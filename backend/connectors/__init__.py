# Agent-callable connectors to external systems (databases, ERPs, comms, ticketing).
# Read-only first (see sql_read.py); write-capable connectors go through the
# agent-executor's write-approval gate (backend/agent/executor.py).
