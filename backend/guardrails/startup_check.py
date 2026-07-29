"""
startup_check.py — self-test suite executed at FastAPI startup.

Ensures critical configurations and connection channels are fully functional.
Blocks server startup on critical failure (Pydantic config validation, regex errors,
Postgres down, normalizer failure).
Logs warning on non-critical failures (Qdrant down — handled by fallback keyword path).
"""
from __future__ import annotations

import logging
import re

from backend.guardrails.config_schema import validate_guardrail_config
from backend.storage.postgres_store import PostgresStore

logger = logging.getLogger(__name__)


def run_startup_self_test(config: dict) -> None:
    """
    Run all startup self-tests.
    Raises RuntimeError on critical failure, preventing server from starting.
    """
    logger.info("Running guardrail startup self-test...")
    errors: list[str] = []

    # ── 1. Config Validation ──────────────────────────────────────────────────
    try:
        validate_guardrail_config(config)
        logger.info("Startup check: configuration schema VALID")
    except Exception as e:
        errors.append(f"[CONFIG] Validation failed: {e}")

    # ── 2. Regex Pattern Compilation ──────────────────────────────────────────
    try:
        from backend.guardrails.input_guard import (
            _GSTIN, _PAN, _AADHAAR, _CREDIT_CARD, _EMAIL, _UPI, _PHONE,
        )
        # compile check is implicit since they are compiled at load time,
        # but let's run a quick match check to ensure engine doesn't throw.
        _GSTIN.search("22AAAAA0000A1Z5")
        _PAN.search("ABCDE1234F")
        _AADHAAR.search("1234-5678-9012")
        _CREDIT_CARD.search("1234-5678-1234-5678")
        _EMAIL.search("test@example.com")
        _UPI.search("test@ybl")
        _PHONE.search("9876543210")
        logger.info("Startup check: all PII regex patterns verified")
    except Exception as e:
        errors.append(f"[REGEX] Pattern error: {e}")

    # ── 3. Postgres Availability ──────────────────────────────────────────────
    try:
        pg = PostgresStore()
        with pg.conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        logger.info("Startup check: PostgreSQL connection OK")
    except Exception as e:
        errors.append(f"[POSTGRES] Database is unreachable: {e}")

    # ── 4. Qdrant Availability (Non-blocking warning) ──────────────────────────
    try:
        from backend.storage.qdrant_store import QdrantStore
        qdr = QdrantStore(config)
        qdr.client.get_collections()
        logger.info("Startup check: Qdrant connection OK")
    except Exception as e:
        logger.warning(
            "[STARTUP CHECK] Qdrant connection check FAILED: %s. "
            "Server will start, but retrieval degradation mode (keyword search) "
            "will be active on vector failures.", e
        )

    # ── 5. Normalizer Smoke Test ──────────────────────────────────────────────
    try:
        from backend.guardrails.normalizer import normalize
        res = normalize("Ｉｇｎｏｒｅ\u200b")
        if res != "Ignore":
            errors.append(
                f"[NORMALIZER] Smoke test failed: expected 'Ignore', got {res!r}"
            )
        else:
            logger.info("Startup check: NFKC Normalizer smoke test OK")
    except Exception as e:
        errors.append(f"[NORMALIZER] Internal error: {e}")

    # ── Verify ────────────────────────────────────────────────────────────────
    if errors:
        msg = (
            "Guardrail startup self-test failed. The server cannot start:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info("Guardrail startup self-test completed successfully.")
