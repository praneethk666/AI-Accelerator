"""
health_probe.py — background health monitoring task.

Regularly tests connections to PostgreSQL and Qdrant collection channels.
Updates OTel gauges indicating service status (1=healthy, 0=down) and logs alerts.
Allows the system to activate degradation pathways immediately rather than waiting
for user requests to crash.
"""
from __future__ import annotations

import asyncio
import logging
from opentelemetry import metrics
from opentelemetry.metrics import Observation

from backend.storage.postgres_store import PostgresStore
from backend.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# Default probe interval
_DEFAULT_PROBE_INTERVAL = 60

# Global status registers so retrieval can query them directly if needed
_HEALTH_STATUS = {
    "postgres": True,
    "qdrant":   True,
}


def _health_callback(options) -> list[Observation]:
    """OTel callback returning health status for Postgres and Qdrant."""
    return [
        Observation(1.0 if _HEALTH_STATUS["postgres"] else 0.0, {"service": "postgres"}),
        Observation(1.0 if _HEALTH_STATUS["qdrant"] else 0.0, {"service": "qdrant"}),
    ]


# Lazy OTel initialization
_meter = metrics.get_meter("ai-accelerator")
_HEALTH_GAUGE = _meter.create_observable_gauge(
    "health_status",
    description="Health status of external services (1=healthy, 0=down)",
    callbacks=[_health_callback]
)


def is_healthy(service: str) -> bool:
    """Returns True if *service* was healthy during the last background probe."""
    return _HEALTH_STATUS.get(service, True)


async def _probe_postgres() -> bool:
    """Check PostgreSQL health. Returns True if database responds."""
    try:
        # Wrap postgres test inside to_thread to avoid blocking event loop
        def run_check():
            pg = PostgresStore()
            with pg.conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            pg.conn.close()
        await asyncio.to_thread(run_check)
        return True
    except Exception as e:
        logger.error("[HEALTH CHECK] Postgres is DOWN: %s", e)
        return False


async def _probe_qdrant(config: dict) -> bool:
    """Check Qdrant health. Returns True if Qdrant responds."""
    try:
        def run_check():
            qdr = QdrantStore(config)
            qdr.client.get_collections()
        await asyncio.to_thread(run_check)
        return True
    except Exception as e:
        logger.error("[HEALTH CHECK] Qdrant is DOWN: %s", e)
        return False


async def background_health_loop(config: dict) -> None:
    """
    Async background task that runs indefinitely.
    Probes external services and logs state changes.
    """
    g_cfg = (config.get("guardrails") or {}).get("health_probe") or {}
    if not g_cfg.get("enabled", True):
        logger.info("Background health probe is disabled in config.")
        return

    interval = int(g_cfg.get("interval_seconds", _DEFAULT_PROBE_INTERVAL))
    logger.info("Starting background health loop (interval=%ds)...", interval)

    while True:
        try:
            # Run probes concurrently
            pg_ok, qdr_ok = await asyncio.gather(
                _probe_postgres(),
                _probe_qdrant(config)
            )

            # Log postgres transition
            if pg_ok != _HEALTH_STATUS["postgres"]:
                _HEALTH_STATUS["postgres"] = pg_ok
                if pg_ok:
                    logger.info("[HEALTH ALERT] Postgres has RECOVERED")
                else:
                    logger.critical("[HEALTH ALERT] Postgres is DOWN")

            # Log qdrant transition
            if qdr_ok != _HEALTH_STATUS["qdrant"]:
                _HEALTH_STATUS["qdrant"] = qdr_ok
                if qdr_ok:
                    logger.info("[HEALTH ALERT] Qdrant has RECOVERED")
                else:
                    logger.critical("[HEALTH ALERT] Qdrant is DOWN")

        except Exception as e:
            logger.warning("background_health_loop error: %s", e)

        await asyncio.sleep(interval)
