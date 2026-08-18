"""
session_manager.py — UUID-based session tracking tied strictly to caller identity.
Implements event history buffer for SSE Last-Event-ID resumability.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class StreamEvent:
    event_id: int
    event_name: str
    data: str
    timestamp: float = field(default_factory=time.time)


class Session:
    def __init__(self, session_id: str, caller_identity: str):
        self.session_id = session_id
        self.caller_identity = caller_identity
        self.created_at = time.time()
        self.last_active_at = time.time()
        self.next_event_id = 1
        self.events: List[StreamEvent] = []
        self.max_event_history = 100
        self.queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

    def add_event(self, event_name: str, data: str) -> StreamEvent:
        event = StreamEvent(
            event_id=self.next_event_id,
            event_name=event_name,
            data=data,
        )
        self.next_event_id += 1
        self.events.append(event)
        if len(self.events) > self.max_event_history:
            self.events.pop(0)
        self.last_active_at = time.time()
        try:
            self.queue.put_nowait(event)
        except Exception:
            pass
        return event

    def get_events_after(self, last_event_id: int) -> List[StreamEvent]:
        return [e for e in self.events if e.event_id > last_event_id]


class SessionManager:
    """Manages active sessions and validates identity ownership."""

    def __init__(self, session_ttl_seconds: int = 3600):
        self._sessions: Dict[str, Session] = {}
        self.session_ttl_seconds = session_ttl_seconds

    def create_session(self, caller_identity: str, session_id: Optional[str] = None) -> Session:
        """Creates and stores a new session tied to the caller identity."""
        sid = session_id or str(uuid.uuid4())
        session = Session(session_id=sid, caller_identity=caller_identity)
        self._sessions[sid] = session
        return session

    def validate_or_bind_session(
        self, session_id: str, caller_identity: str
    ) -> Tuple[bool, Optional[Session], Optional[str]]:
        """
        Validates session ownership.
        Returns (is_valid, session_obj, error_message).
        - If session exists and belongs to caller_identity: Valid.
        - If session exists but belongs to a DIFFERENT caller: Denied (Identity mismatch).
        - If session does not exist: Creates a new session for this caller.
        """
        session = self._sessions.get(session_id)
        if session is not None:
            # Check expiry
            if time.time() - session.last_active_at > self.session_ttl_seconds:
                # Expired session -> replace with new session
                del self._sessions[session_id]
                session = self.create_session(caller_identity, session_id=session_id)
                return True, session, None

            # Enforce identity binding
            if session.caller_identity != caller_identity:
                return (
                    False,
                    None,
                    f"Session '{session_id}' is owned by another identity. Access denied.",
                )

            session.last_active_at = time.time()
            return True, session, None

        # If session ID is provided but unknown, create new session bound to this caller
        new_session = self.create_session(caller_identity, session_id=session_id)
        return True, new_session, None

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def clean_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_active_at > self.session_ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]


# Global singleton instance
session_manager = SessionManager()
