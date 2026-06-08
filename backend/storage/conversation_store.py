"""Conversation store interface + stub.

Karthii owns the implementation (PostgreSQL-backed).
Abhishek's AnswerTool calls save_turn() and load_history() — import from here.

Do not change the function signatures without telling both Karthii and Abhishek.
"""
from __future__ import annotations
from typing import Protocol


class ConversationStore(Protocol):
    def save_turn(self, session_id: str, role: str, content: str) -> None:
        """Append one turn to the conversation history."""
        ...

    def load_history(self, session_id: str, n: int = 10) -> list[dict]:
        """Return the last n turns as [{"role": ..., "content": ...}, ...]."""
        ...


def get_conversation_store() -> ConversationStore:
    """Return the active store. Karthii replaces the body with the real implementation."""
    raise NotImplementedError("Karthii: implement this in feat/karthii-pipeline-skeleton")
