"""request_clarification — a CONTROL tool the agent calls to ask the USER to choose.

Unlike the other agent tools this one has no side effect and is never really
dispatched: the executor INTERCEPTS it (like the write-approval gate), pauses the
loop, and returns status="needs_clarification" with the question + options so the
UI can render a chooser instead of the agent guessing. `run` exists only to satisfy
the AgentTool protocol and as a harmless fallback if it is ever dispatched directly.
"""
from __future__ import annotations


class RequestClarificationTool:
    name = "request_clarification"
    description = (
        "Ask the USER to clarify or choose when the request is ambiguous — e.g. "
        "several documents match and you cannot tell which one they mean. ALWAYS "
        "prefer this over guessing. The conversation pauses and the user is shown "
        "your question and options as buttons; their choice returns as the next "
        "message. Do NOT use this for questions the documents can answer — search first."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The single clear question to ask the user.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-8 concrete choices (e.g. candidate filenames). "
                "Omit for a free-text question.",
            },
        },
        "required": ["question"],
    }

    def run(self, question: str, options=None) -> dict:
        # Intercepted by the executor before dispatch; safety-net only.
        return {"question": question, "options": options or []}

    __call__ = run
