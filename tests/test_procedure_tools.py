"""Tests for StartProcedureWalkthroughTool / AdvanceProcedureStepTool --
agent-driven guided procedure walkthroughs (backend/agent/procedure_tools.py),
state persisted via backend/storage/conversation_store.py's
get/set_session_active_procedure. All Postgres calls mocked here; the real
live-DB round trip is covered separately in tests/test_conversation_store.py.

Config-gated off by default (query.agent.procedure_walkthrough.enabled), same
"clear error, never silent" posture as browse_by_equipment/browse_document_outline.
"""
from unittest.mock import MagicMock, patch

from backend.agent.procedure_tools import AdvanceProcedureStepTool, StartProcedureWalkthroughTool

_ENABLED_CFG = {"query": {"agent": {"procedure_walkthrough": {"enabled": True, "max_steps": 50}}}}
_DISABLED_CFG = {"query": {"agent": {"procedure_walkthrough": {"enabled": False}}}}
_UNSET_CFG = {"query": {"agent": {}}}

_PARSED = {
    "section_title": "1.1 Replacing the Workpiece Holder",
    "page_range": [5, 5],
    "first_step": "1",
    "steps": {
        "1": {"text": "Place the switch to MANU.", "page": 5, "next": "2"},
        "2": {"text": "Press the MASTER ON button.", "page": 5, "next": "3"},
        "3": {"text": "Advance both centers.", "page": 5, "next": None},
    },
}


def _fake_pg(blocks=None):
    pg = MagicMock()
    pg.get_blocks.return_value = blocks if blocks is not None else [{"type": "text", "text": "x"}]
    return pg


# ── StartProcedureWalkthroughTool ─────────────────────────────────────────────

def test_start_missing_document_id_returns_error():
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG):
        result = StartProcedureWalkthroughTool().run(document_id="", session_id="s1")
    assert "error" in result


def test_start_no_locator_returns_error():
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG):
        result = StartProcedureWalkthroughTool().run(document_id="d1", session_id="s1")
    assert "error" in result


def test_start_no_session_id_returns_error():
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG):
        result = StartProcedureWalkthroughTool().run(document_id="d1", section_title="X", session_id=None)
    assert "error" in result


def test_start_disabled_by_default_returns_clear_error():
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_DISABLED_CFG):
        result = StartProcedureWalkthroughTool().run(document_id="d1", section_title="X", session_id="s1")
    assert "error" in result
    assert "disabled" in result["error"].lower()


def test_start_disabled_when_config_key_entirely_absent():
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_UNSET_CFG):
        result = StartProcedureWalkthroughTool().run(document_id="d1", section_title="X", session_id="s1")
    assert "error" in result


def test_start_no_blocks_returns_error():
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_pg(blocks=[])):
        result = StartProcedureWalkthroughTool().run(document_id="d1", section_title="X", session_id="s1")
    assert "error" in result


def test_start_parse_failure_returns_clear_fallback_error():
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_pg()), \
         patch("backend.agent.step_parser.parse_procedure_from_blocks", return_value=None):
        result = StartProcedureWalkthroughTool().run(document_id="d1", section_title="X", session_id="s1")
    assert "error" in result
    assert "normal prose" in result["error"]


def test_start_success_writes_state_and_returns_step_one():
    store = MagicMock()
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_pg()), \
         patch("backend.agent.step_parser.parse_procedure_from_blocks", return_value=_PARSED), \
         patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        result = StartProcedureWalkthroughTool().run(
            document_id="d1", section_title="Replacing the Workpiece Holder", session_id="s1")

    assert result["step_id"] == "1"
    assert result["step_text"] == "Place the switch to MANU."
    assert result["has_next"] is True
    assert result["status"] == "in_progress"

    store.set_session_active_procedure.assert_called_once()
    sid, procedure = store.set_session_active_procedure.call_args.args
    assert sid == "s1"
    assert procedure["document_id"] == "d1"
    assert procedure["current_step"] == "1"
    assert procedure["status"] == "in_progress"


def test_start_passes_configured_max_steps_to_parser():
    cfg = {"query": {"agent": {"procedure_walkthrough": {"enabled": True, "max_steps": 12}}}}
    with patch("backend.agent.procedure_tools._load_default_config", return_value=cfg), \
         patch("backend.storage.postgres_store.PostgresStore", return_value=_fake_pg()), \
         patch("backend.agent.step_parser.parse_procedure_from_blocks", return_value=_PARSED) as mock_parse, \
         patch("backend.storage.conversation_store.PostgresConversationStore"):
        StartProcedureWalkthroughTool().run(document_id="d1", section_title="X", session_id="s1")
    assert mock_parse.call_args.kwargs["max_steps"] == 12


# ── AdvanceProcedureStepTool ───────────────────────────────────────────────────

def _procedure(current="2"):
    return {
        "document_id": "d1", "section_title": "1.1 Replacing the Workpiece Holder",
        "page_range": [5, 5], "current_step": current, "status": "in_progress",
        "steps": {
            "1": {"text": "Place the switch to MANU.", "page": 5, "next": "2"},
            "2": {"text": "Press the MASTER ON button.", "page": 5, "next": "3"},
            "3": {"text": "Advance both centers.", "page": 5, "next": None},
        },
    }


def test_advance_no_session_id_returns_error():
    result = AdvanceProcedureStepTool().run(action="next", session_id=None)
    assert "error" in result


def test_advance_unknown_action_returns_error():
    result = AdvanceProcedureStepTool().run(action="bogus", session_id="s1")
    assert "error" in result


def test_advance_disabled_returns_clear_error():
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_DISABLED_CFG):
        result = AdvanceProcedureStepTool().run(action="next", session_id="s1")
    assert "error" in result
    assert "disabled" in result["error"].lower()


def test_advance_no_active_procedure_returns_error():
    store = MagicMock()
    store.get_session_active_procedure.return_value = None
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        result = AdvanceProcedureStepTool().run(action="next", session_id="s1")
    assert "error" in result
    assert "start_procedure_walkthrough" in result["error"]


def test_advance_next_moves_to_next_step():
    store = MagicMock()
    store.get_session_active_procedure.return_value = _procedure(current="1")
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        result = AdvanceProcedureStepTool().run(action="next", session_id="s1")
    assert result["step_id"] == "2"
    assert result["step_text"] == "Press the MASTER ON button."
    store.set_session_active_procedure.assert_called_once()
    _, saved = store.set_session_active_procedure.call_args.args
    assert saved["current_step"] == "2"


def test_advance_next_on_last_step_completes_and_clears_state():
    store = MagicMock()
    store.get_session_active_procedure.return_value = _procedure(current="3")
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        result = AdvanceProcedureStepTool().run(action="next", session_id="s1")
    assert result["status"] == "completed"
    store.set_session_active_procedure.assert_called_once_with("s1", None)


def test_advance_repeat_returns_current_step_without_state_change():
    store = MagicMock()
    store.get_session_active_procedure.return_value = _procedure(current="2")
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        result = AdvanceProcedureStepTool().run(action="repeat", session_id="s1")
    assert result["step_id"] == "2"
    store.set_session_active_procedure.assert_not_called()


def test_advance_back_returns_to_previous_step():
    store = MagicMock()
    store.get_session_active_procedure.return_value = _procedure(current="2")
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        result = AdvanceProcedureStepTool().run(action="back", session_id="s1")
    assert result["step_id"] == "1"


def test_advance_back_on_first_step_returns_error():
    store = MagicMock()
    store.get_session_active_procedure.return_value = _procedure(current="1")
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        result = AdvanceProcedureStepTool().run(action="back", session_id="s1")
    assert "error" in result


def test_advance_goto_jumps_to_target_step():
    store = MagicMock()
    store.get_session_active_procedure.return_value = _procedure(current="1")
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        result = AdvanceProcedureStepTool().run(action="goto", step_id="3", session_id="s1")
    assert result["step_id"] == "3"


def test_advance_goto_missing_step_id_returns_error():
    store = MagicMock()
    store.get_session_active_procedure.return_value = _procedure(current="1")
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        result = AdvanceProcedureStepTool().run(action="goto", session_id="s1")
    assert "error" in result


def test_advance_goto_unknown_step_id_returns_error():
    store = MagicMock()
    store.get_session_active_procedure.return_value = _procedure(current="1")
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        result = AdvanceProcedureStepTool().run(action="goto", step_id="99", session_id="s1")
    assert "error" in result


def test_advance_abort_clears_state():
    store = MagicMock()
    store.get_session_active_procedure.return_value = _procedure(current="2")
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        result = AdvanceProcedureStepTool().run(action="abort", session_id="s1")
    assert result["status"] == "aborted"
    store.set_session_active_procedure.assert_called_once_with("s1", None)


def test_advance_exposes_branches_when_present():
    proc = _procedure(current="1")
    proc["steps"]["1"]["branches"] = [{"condition": "the light is red", "next": "5"}]
    store = MagicMock()
    store.get_session_active_procedure.return_value = proc
    with patch("backend.agent.procedure_tools._load_default_config", return_value=_ENABLED_CFG), \
         patch("backend.storage.conversation_store.PostgresConversationStore", return_value=store):
        result = AdvanceProcedureStepTool().run(action="repeat", session_id="s1")
    assert result["branches"] == [{"condition": "the light is red", "next": "5"}]
