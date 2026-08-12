"""backend/pipeline/quality_gates.py — Ingestion quality gate (P1).

Runs AFTER run_pipeline(), BEFORE finalize_document(). Produces a
`ingestion_quality_score` (0..1) and a list of quality warnings.

Three checks:
  1. chunk_count_sanity  — flag if new count dropped >30% vs. previous
  2. confidence_score    — average extraction_confidence across chunks
  3. part_number_check   — warn if a CAD/BOM chunk has zero component codes

The score is a weighted average of sub-scores; thresholds are configurable
under config.ingestion.quality_gates in global.yaml:

    ingestion:
      quality_gates:
        enabled: true
        chunk_drop_threshold: 0.30   # flag if chunk count drops more than 30%
        min_quality_score: 0.50      # warn (non-fatal) if score < this

Usage:
    from backend.pipeline.quality_gates import run_quality_gates

    quality = run_quality_gates(result, previous_chunk_count=42, config=cfg)
    # quality = {"score": 0.87, "warnings": [...], "checks": {...}}
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ─── defaults ────────────────────────────────────────────────────────────────

_DEFAULT_CHUNK_DROP_THRESHOLD = 0.30   # 30% drop → flag
_DEFAULT_MIN_QUALITY_SCORE    = 0.50   # warn below this


def run_quality_gates(
    result: dict,
    previous_chunk_count: int | None = None,
    config: dict | None = None,
) -> dict:
    """Evaluate ingestion quality after the pipeline finishes.

    Args:
        result:               The dict returned by run_pipeline().
        previous_chunk_count: The chunk_count from the document row BEFORE
                              this re-ingestion started (None for first ingest).
        config:               Full config dict (reads ingestion.quality_gates).

    Returns:
        {
            "score":    float,          # 0..1 composite quality score
            "warnings": list[str],      # human-readable warnings (non-fatal)
            "checks":   dict,           # per-check sub-scores + details
        }
    """
    qg_cfg = (config or {}).get("ingestion", {}).get("quality_gates", {})
    enabled = qg_cfg.get("enabled", True)

    if not enabled:
        return {"score": 1.0, "warnings": [], "checks": {"enabled": False}}

    drop_threshold = float(qg_cfg.get("chunk_drop_threshold", _DEFAULT_CHUNK_DROP_THRESHOLD))
    min_quality    = float(qg_cfg.get("min_quality_score",    _DEFAULT_MIN_QUALITY_SCORE))

    chunks: list[dict]  = result.get("chunks") or []
    warnings: list[str] = []
    checks:   dict      = {}

    # ── Check 1: chunk count sanity ──────────────────────────────────────────
    count_score = 1.0
    new_count = len(chunks)
    checks["chunk_count"] = {"new": new_count, "previous": previous_chunk_count}

    if previous_chunk_count and previous_chunk_count > 0 and new_count > 0:
        drop_frac = (previous_chunk_count - new_count) / previous_chunk_count
        checks["chunk_count"]["drop_fraction"] = round(drop_frac, 3)
        if drop_frac > drop_threshold:
            msg = (
                f"Chunk count dropped {drop_frac:.0%} "
                f"({previous_chunk_count} → {new_count}). "
                f"Threshold: {drop_threshold:.0%}. "
                "Check if extraction quality degraded or pages were removed."
            )
            warnings.append(msg)
            logger.warning("quality_gates: %s", msg)
            # Penalise score proportionally — a 50% drop → 0.0 score contribution
            count_score = max(0.0, 1.0 - (drop_frac / drop_threshold) * 0.5)
    elif new_count == 0:
        warnings.append("Zero chunks produced — document may be empty or extraction failed.")
        count_score = 0.0
    checks["chunk_count"]["score"] = round(count_score, 3)

    # ── Check 2: extraction confidence average ───────────────────────────────
    confidences = []
    for c in chunks:
        tags = c.get("tags") or {}
        conf = tags.get("extraction_confidence")
        if conf is not None:
            try:
                confidences.append(float(conf))
            except (TypeError, ValueError):
                pass

    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        conf_score = avg_conf  # already 0..1
    else:
        avg_conf  = None
        conf_score = 1.0  # no confidence data → don't penalise (most tools don't set it)

    checks["extraction_confidence"] = {
        "average": round(avg_conf, 3) if avg_conf is not None else None,
        "n_chunks_with_conf": len(confidences),
        "score": round(conf_score, 3),
    }

    # ── Check 3: CAD/BOM component code coverage ─────────────────────────────
    part_score  = 1.0
    cad_chunks  = [c for c in chunks if (c.get("tags") or {}).get("doc_type") in ("cad", "bom")]
    no_code_cnt = sum(
        1 for c in cad_chunks
        if not (c.get("tags") or {}).get("component_codes")
    )
    if cad_chunks:
        missing_frac = no_code_cnt / len(cad_chunks)
        part_score   = 1.0 - missing_frac
        checks["component_codes"] = {
            "cad_chunks": len(cad_chunks),
            "missing_codes": no_code_cnt,
            "score": round(part_score, 3),
        }
        if missing_frac > 0.5:
            warnings.append(
                f"{no_code_cnt}/{len(cad_chunks)} CAD/BOM chunks have no component codes."
            )
    else:
        checks["component_codes"] = {"cad_chunks": 0, "score": 1.0}

    # ── Composite score (weighted average) ───────────────────────────────────
    # chunk_count carries the most weight (data completeness); confidence next.
    score = round(
        0.50 * count_score +
        0.30 * conf_score  +
        0.20 * part_score,
        3,
    )
    checks["composite_score"] = score

    if score < min_quality:
        warnings.append(
            f"Ingestion quality score {score:.2f} is below minimum threshold {min_quality:.2f}. "
            "Consider reviewing extraction settings or re-ingesting with a different pipeline."
        )
        logger.warning(
            "quality_gates: score=%.2f below min=%.2f for this ingest run",
            score, min_quality,
        )

    logger.info(
        "quality_gates: score=%.2f chunks=%d prev=%s warnings=%d",
        score, new_count, previous_chunk_count, len(warnings),
    )

    return {"score": score, "warnings": warnings, "checks": checks}


def compute_index_version(model_name: str, config: dict | None = None) -> str:
    """Derive a stable index_version string from the embedding model name and date.

    Format: ``<model_basename>@<YYYY-MM-DD>``
    Example: ``bge-m3@2026-08-11``

    Also includes a short config hash so two runs with the same model but
    different embedding configs are distinguishable.
    """
    import datetime
    import hashlib

    date_str = datetime.date.today().isoformat()
    base     = model_name.split("/")[-1]          # strip org prefix

    # Hash the relevant embeddings config (dense_dim + model) for disambiguation.
    emb_cfg   = (config or {}).get("embeddings", {})
    cfg_str   = f"{emb_cfg.get('model', '')}:{emb_cfg.get('dense_dim', '')}"
    cfg_short = hashlib.sha256(cfg_str.encode()).hexdigest()[:8]

    return f"{base}@{date_str}:{cfg_short}"
