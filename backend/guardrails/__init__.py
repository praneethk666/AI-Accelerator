"""
backend/guardrails — production safety layer for the AI Accelerator RAG pipeline.

Three checkpoints around the LangGraph agent loop:
  1. input_guard_node  — runs BEFORE agent_node (injection detection, PII redact in query)
  2. scan_tool_output  — runs inside tools_node on every tool result (recursive JSON scan)
  3. output_guard_node — runs LAST before END (PII mask in answer, DB gets safe_answer only)

Every check is wrapped by @guardrail_safe — a guard crash never crashes the request.
"""
