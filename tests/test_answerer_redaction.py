"""Tests for AnswererTool surfacing redacted source content explicitly instead
of silently hallucinating or echoing literal asterisks back with no explanation.
Real finding, 3-Aug: a CAD sheet's own parts table had its values blanked out
("***") in the source file -- backend/categorize/redaction_detect.py flags it at
ingest time, chunk_tool.py propagates the flag into the chunk, and this is where
the answer prompt actually gets told about it.

Mock-only (LLM), matches tests/test_answerer_image_ground.py's style.
"""
from unittest.mock import MagicMock, patch

from backend.retrieval.answerer import AnswererTool


def _fake_llm(response_text="the answer"):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=response_text)
    return llm


def _chunk(chunk_id, text, redacted=False, redaction_reason=None):
    c = {
        "chunk_id": chunk_id, "document_id": "doc-1", "text": text,
        "token_count": 200, "tags": {"summary": "ok"},   # non-thin: real summary +
                                                         # >= _THIN_TOKEN_FLOOR --
                                                         # avoids the thin-chunk DB
                                                         # expansion path entirely
        "source_ref": {"page": 1, "filename": "spindle.pdf"},
    }
    if redacted:
        c["redacted"] = True
        c["redaction_reason"] = redaction_reason
    return c


def _state(chunks):
    return {"query": "what is the part number?", "retrieved_chunks": chunks,
            "session_id": "", "conversation_history": [], "errors": []}


def _config():
    return {"llm": {"answer_model": "gpt-4o-mini"}, "query": {"answerer": {}}}


def test_redacted_chunk_gets_explicit_note_in_prompt():
    chunks = [_chunk("c1", "| No. | Parts No. |\n| *** | **-*-* |",
                     redacted=True, redaction_reason="Values are blanked out in the source.")]
    llm = _fake_llm()
    with patch("backend.retrieval.answerer.get_llm_for", return_value=llm), \
         patch("backend.retrieval.answerer.resolve_model_provider",
               return_value=("gpt-4o-mini", "openai")):
        AnswererTool().run(_state(chunks), _config())

    user_msg = llm.invoke.call_args[0][0][1]["content"]
    assert "REDACTED IN SOURCE" in user_msg
    assert "Values are blanked out in the source." in user_msg


def test_non_redacted_chunk_gets_no_note():
    chunks = [_chunk("c1", "Torque to 12 N*m.")]
    llm = _fake_llm()
    with patch("backend.retrieval.answerer.get_llm_for", return_value=llm), \
         patch("backend.retrieval.answerer.resolve_model_provider",
               return_value=("gpt-4o-mini", "openai")):
        AnswererTool().run(_state(chunks), _config())

    user_msg = llm.invoke.call_args[0][0][1]["content"]
    assert "REDACTED IN SOURCE" not in user_msg
