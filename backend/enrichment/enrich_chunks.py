"""EnrichChunksTool — stamp category tags + a short summary + meaningful keywords
on each chunk.

  run(state, config)
    READS  state["chunks"], state["industry"], state["document_type"]
    WRITES chunk["tags"]  (industry, doc_type, section, summary, keywords)

Enrichment is LLM-based (default): chunks are processed in BATCHES — one LLM call
describes several chunks at once and returns a JSON array of {summary, keywords}.
Batching keeps the call count low (e.g. 43 chunks -> ~5 calls) so we don't trip
provider rate limits, which previously made calls fail silently and fall back to
meaningless frequency words. If a batch still fails (after one retry) those chunks
fall back to frequency keywords and the count is surfaced in state["errors"].
Set enrichment.summarize: false for pure offline frequency keywords.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter

from backend.core.tool import PipelineState

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "with", "as", "by", "at", "from", "is", "are", "was", "were", "be",
    "this", "that", "these", "those", "it", "its", "we", "you", "they", "i",
    "not", "no", "can", "will", "shall", "may", "must", "should", "would", "when",
    "what", "which", "into", "your", "their", "have", "has", "had", "also",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")

# Min seconds between enrichment LLM calls (free-tier RPM pacing). Set per run from
# config in run(); read by _enrich_batch which is the single LLM call site.
_ENRICH_INTERVAL_S = 0.0

# Instruction describing WHAT to produce per chunk. Editable via config
# enrichment.prompt / the Settings UI. The code wraps it with the numbered chunks
# and asks for a JSON array, so the instruction should NOT include a chunk itself.
_DEFAULT_INSTRUCTION = (
    "You label document chunks for a technical search index. For EACH chunk produce:\n"
    "- summary: one precise sentence stating what the chunk is about and what a reader "
    "would learn from it, in the document's own terms.\n"
    "- keywords: 6-10 SPECIFIC, high-value search terms taken from the chunk — include "
    "part numbers, model names, component/procedure names, specs and units, and domain "
    "terms. Never include stopwords, generic words (e.g. 'information', 'section', "
    "'manual'), or terms not present in the chunk."
)


class EnrichChunksTool:
    name = "enrich_chunks"

    def run(self, state: PipelineState, config: dict) -> PipelineState:
        industry = state.get("industry")
        doc_type = state.get("document_type")
        ecfg = config.get("enrichment", {}) or {}
        top_k = ecfg.get("keyword_count", 6)
        summarize = ecfg.get("summarize", True)
        batch_size = max(1, int(ecfg.get("batch_size", 10)))
        global _ENRICH_INTERVAL_S
        _ENRICH_INTERVAL_S = float(ecfg.get("min_interval_s", 0) or 0)
        instruction = (ecfg.get("prompt") or _DEFAULT_INSTRUCTION).strip()
        chunks = state.get("chunks", [])

        llm = None
        if summarize and chunks:
            try:
                from backend.core.llm_client import get_llm_for
                # Size the completion budget to the batch so the JSON array for the
                # whole batch fits: ~160 tokens/chunk (one-line summary + keywords +
                # JSON punctuation) plus a fixed buffer. This stops the array being
                # truncated mid-object; split-retry below handles any residual case.
                max_out = 160 * batch_size + 256
                # per-step override: enrichment.model (bulk/cheap) falls back to global llm.
                llm = get_llm_for(config, config.get("enrichment"), max_tokens=max_out)
            except Exception:
                llm = None

        results: dict[int, tuple] = {}
        eligible_indices: set[int] = set()
        if llm is not None:
            from backend.core import usage
            eligible = [
                (i, (c.get("text") or "").strip())
                for i, c in enumerate(chunks)
                if len((c.get("text") or "").strip()) >= 20
            ]
            eligible_indices = {i for i, _ in eligible}
            for start in range(0, len(eligible), batch_size):
                from backend.core.tool import check_cancelled
                check_cancelled(state.get("document_id"))
                _enrich_group(llm, instruction,
                              eligible[start:start + batch_size], usage, results)

        # Two DIFFERENT reasons a chunk ends up with frequency-only keywords, not one:
        # too_short chunks were never sent to the LLM at all (by design — not worth a
        # call for a bare heading like "3 Wiring"); llm_failed chunks WERE sent but
        # the batch never produced a result for them (a real quality loss, worth
        # surfacing). Conflating them under one "fallback" count previously read as
        # an 18-23% failure rate on a real run that had ZERO actual LLM failures —
        # every one was a legitimately-short heading. Only llm_failed is an error.
        too_short = 0
        llm_failed = 0
        for i, chunk in enumerate(chunks):
            tags = chunk.setdefault("tags", {})
            if industry is not None:
                tags.setdefault("industry", industry)
            if doc_type is not None:
                tags.setdefault("doc_type", doc_type)
            ref = chunk.get("source_ref") or {}
            section = ref.get("section") if isinstance(ref, dict) else None
            if section:
                tags.setdefault("section", section)

            summary, kws = results.get(i, (None, None))
            if summary:
                tags["summary"] = summary
            if kws:
                tags["keywords"] = kws
            else:
                tags["keywords"] = _keywords(chunk.get("text") or "", top_k)
                if llm is not None:
                    if i in eligible_indices:
                        llm_failed += 1
                    else:
                        too_short += 1

        if too_short:
            logger.info("enrich: %d/%d chunks too short for LLM summarization "
                        "(used frequency keywords instead — expected, not an error)",
                        too_short, len(chunks))
        if llm is not None and llm_failed:
            state.setdefault("errors", []).append(
                f"enrich: {llm_failed}/{len(chunks)} chunks used fallback keywords "
                f"(LLM enrichment failed for those batches)"
            )
        return state


def _store(group, data, results) -> None:
    """Map a JSON array (aligned by order) onto the chunk indices in `group`."""
    for (idx, _), item in zip(group, data):
        if isinstance(item, dict):
            kws = item.get("keywords")
            results[idx] = (
                (item.get("summary") or "").strip() or None,
                kws if isinstance(kws, list) and kws else None,
            )


def _enrich_group(llm, instruction: str, group: list, usage, results: dict) -> None:
    """Enrich one group of chunks and INSIST on a complete reply.

    If the model returns fewer objects than chunks (the classic symptom of the JSON
    array being truncated at the output-token limit), we do NOT keep the half — we
    split the group and re-ask each half, recursing down to single chunks. A
    one-chunk request can't overflow the token budget, so this guarantees every
    chunk gets a whole response (or, only if a single chunk fails twice, a clean
    frequency fallback in run()). No mangled partial arrays."""
    if not group:
        return
    texts = [t for _, t in group]
    data = _enrich_batch(llm, instruction, texts, usage)
    if data and len(data) >= len(group):
        _store(group, data, results)
        return
    if len(group) == 1:
        # smallest possible request; one more attempt, else leave for the fallback.
        data = _enrich_batch(llm, instruction, texts, usage)
        if data:
            _store(group, data, results)
        return
    mid = len(group) // 2
    _enrich_group(llm, instruction, group[:mid], usage, results)
    _enrich_group(llm, instruction, group[mid:], usage, results)


def _retry_after_s(msg: str, default: float) -> float:
    """Parse the server's suggested wait ('try again in 2.52s') from a 429 body."""
    m = re.search(r"try again in ([\d.]+)\s*s", msg or "")
    return (float(m.group(1)) + 0.5) if m else default


_RETRYABLE = ("429", "quota", "rate limit", "rate_limit", "ratelimit",
              "resource_exhausted", "resourceexhausted",
              "500", "internal error", "503", "unavailable", "overloaded")


def _is_rate_limit(exc: Exception) -> bool:
    """Matches ANY transient error worth retrying, not just 429s — real fallback
    fell back on 503 UNAVAILABLE errors this narrower check used to miss (validated
    live, 21-Jul: 18-23% of chunks across two documents fell back to offline
    frequency keywords; vision_client.py's broader _RETRYABLE already covered this
    for vision calls, enrichment's own check just hadn't kept up)."""
    s = (str(exc) + type(exc).__name__).lower()
    return any(tok in s for tok in _RETRYABLE)


def _invoke_with_backoff(llm, prompt: str, attempts: int = 4):
    """Invoke the LLM, REACTIVELY backing off on 429s AND transient 5xx errors
    (honoring the server's retry hint when it gives one, e.g. Groq's rate-limit
    body). A batch can tip over a provider's per-minute quota even when request
    pacing is fine, or hit a transient 503 under sustained load — waiting and
    retrying turns that failure into a success instead of a frequency-keyword
    fallback."""
    delay = 2.0
    for i in range(attempts):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            if not _is_rate_limit(exc) or i == attempts - 1:
                raise
            wait = _retry_after_s(str(exc), delay)
            logger.info("enrich: rate-limited; waiting %.1fs then retrying (%d/%d)",
                        wait, i + 1, attempts - 1)
            time.sleep(wait)
            delay = min(delay * 2, 30.0)


def _enrich_batch(llm, instruction: str, texts: list[str], usage) -> list | None:
    """One LLM call describing several chunks; returns a JSON array aligned to texts."""
    parts = [
        instruction,
        "\nReturn ONLY a JSON array with exactly one object per chunk below, in the "
        "same order. Each object: {\"summary\": \"...\", \"keywords\": [\"...\"]}. "
        "No other text.",
    ]
    for idx, t in enumerate(texts):
        parts.append(f'\n--- Chunk {idx} ---\n{t[:1500]}')
    prompt = "\n".join(parts)
    try:
        # Pace under the free-tier RPM (e.g. Groq ~30/min -> ~2s); covers split-retry
        # calls too since every LLM call funnels through here.
        from backend.core import pacing
        pacing.pace("enrichment", _ENRICH_INTERVAL_S)
        reply = _invoke_with_backoff(llm, prompt)
        um = getattr(reply, "usage_metadata", None) or {}
        raw = getattr(reply, "content", str(reply))
        model = getattr(llm, "model", None) or getattr(llm, "model_name", None)
        provider = type(llm).__name__
        usage.record("enrichment", um.get("input_tokens"), um.get("output_tokens"),
                     prompt=prompt, raw_response=raw, provider=provider, model=model)
        return _extract_json_array(raw)
    except Exception as exc:
        # Log WHY so a failed run is diagnosable: a 429/rate-limit reads very
        # differently from a JSON/parse error here.
        logger.warning("enrich batch (%d chunks) failed: %s: %s",
                       len(texts), type(exc).__name__, exc)
        return None


def _keywords(text: str, k: int) -> list[str]:
    """Frequency-ranked content words — offline fallback when the LLM is off/failed."""
    counts = Counter(
        w.lower() for w in _WORD.findall(text) if w.lower() not in _STOPWORDS
    )
    return [word for word, _ in counts.most_common(k)]


def _extract_json_array(text):
    """Pull a JSON array out of an LLM reply that may include prose, code fences,
    or a TRUNCATED array (output hit max tokens). Falls back to decoding whatever
    complete {...} objects are present, in order, so a cut-off reply still yields
    its first N items instead of nothing."""
    t = (text or "").strip()
    if t.startswith("```"):                      # strip code fences
        t = t.strip("`")
        nl = t.find("\n")
        if nl != -1 and t[:nl].strip().lower() in ("json", ""):
            t = t[nl + 1:]
    try:
        out = json.loads(t)
        if isinstance(out, list):
            return out
    except Exception:
        pass
    start, end = t.find("["), t.rfind("]")
    if 0 <= start < end:
        try:
            out = json.loads(t[start:end + 1])
            if isinstance(out, list):
                return out
        except Exception:
            pass
    objs = _decode_objects(t)                     # salvage from a truncated array
    return objs or None


def _decode_objects(t: str) -> list:
    """Decode consecutive top-level JSON objects from text, skipping separators.
    Tolerates a trailing object cut off mid-stream (e.g. truncated output)."""
    dec = json.JSONDecoder()
    objs: list = []
    i, n = 0, len(t)
    while i < n:
        if t[i] == "{":
            try:
                obj, end = dec.raw_decode(t, i)
                objs.append(obj)
                i = end
                continue
            except Exception:
                pass
        i += 1
    return objs
