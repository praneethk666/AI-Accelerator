"""Tests for Dynamic Micro-LLM Procedure Intent Classification, Active Procedure Context Agent, and Tiered Search Router.

Verifies:
1. _has_procedural_step_markers accurately detects step patterns in chunk texts.
2. _classify_procedure_intent_llm uses micro-LLM to evaluate procedure intent.
3. Informational queries with numbered lists (e.g. alarm codes) are classified as NO and return direct answers.
4. Procedural queries with operational steps are classified as YES and offer guided procedures.
5. In active procedure, progress questions ("how many steps completed") are answered directly by the Procedure Agent from memory without searching documents.
6. In active procedure, out-of-procedure questions trigger scoped manual search and prompt for global search on refusal.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from backend.agent.executor import (
    _has_procedural_step_markers,
    _classify_procedure_intent_llm,
    run_agent,
)


class MockStore:
    def __init__(self):
        self.state = {}
        self.proc_cache = {}
        self.ctx_docs: dict[str, list[str]] = {}
    def get_interactive_state(self, session_id):
        return self.state.get(session_id)
    def set_interactive_state(self, session_id, state):
        self.state[session_id] = state
    def get_procedure_cache(self, session_id):
        return self.proc_cache.get(session_id)
    def set_procedure_cache(self, session_id, val):
        self.proc_cache[session_id] = val
    def get_context_docs(self, session_id):
        return self.ctx_docs.get(session_id, [])
    def add_context_doc(self, session_id, doc_id):
        lst = self.ctx_docs.setdefault(session_id, [])
        if doc_id not in lst:
            lst.append(doc_id)


def test_has_procedural_step_markers():
    # Structural step patterns (must return True)
    assert _has_procedural_step_markers("Step 1: Turn off power\nStep 2: Loosen bolts")
    assert _has_procedural_step_markers("(1) Remove cover. (2) Clean the flange.")
    assert _has_procedural_step_markers("1. First step\n2. Second step")
    assert _has_procedural_step_markers("1.1 Preparation\n1.2 Disassembly")
    assert _has_procedural_step_markers("Standard operating procedure for maintenance")
    assert _has_procedural_step_markers("Pre-flight checklist")

    # Plain narrative text without steps (must return False)
    assert not _has_procedural_step_markers("This machine uses an AC servo motor with 200V rating.")
    assert not _has_procedural_step_markers("Dimensions: 500mm x 300mm x 200mm. Weight is 45kg.")
    assert not _has_procedural_step_markers("Hello, how can I help you today?")


def test_classify_procedure_intent_llm():
    mock_llm_yes = MagicMock()
    mock_llm_yes.invoke.return_value = AIMessage(content="YES")
    assert _classify_procedure_intent_llm(
        "clean wheel mounting section",
        "(1) Remove cover. (2) Wipe flange.",
        mock_llm_yes
    )

    mock_llm_no = MagicMock()
    mock_llm_no.invoke.return_value = AIMessage(content="NO")
    assert not _classify_procedure_intent_llm(
        "tell me about 13 h alarm code",
        "Alarm 13H: Inverter overload. 1. Check wiring. 2. Verify power.",
        mock_llm_no
    )


def test_informational_query_does_not_hijack_into_procedure():
    """Verify that an informational query (e.g. alarm code) whose answer contains

    numbered points does NOT trigger procedure_offer when micro-LLM outputs NO.
    """
    config = {
        "query": {
            "agent": {
                "max_iterations": 2,
                "write_tools": [],
                "clarify_tools": ["request_clarification"]
            }
        },
        "guardrails": {"enabled": False}
    }
    
    mock_store = MockStore()
    
    responses = [
        AIMessage(content="Alarm 13H indicates an inverter overload.\n1. Check motor wiring.\n2. Verify power supply."),
        AIMessage(content="NO"),
    ]
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = responses
    mock_llm.bind_tools.return_value = mock_llm

    with patch("backend.storage.conversation_store.get_conversation_store", return_value=mock_store):
        res = run_agent(
            "tell me about 13 h alarm code",
            config=config,
            registry={},
            llm=mock_llm,
            session_id="test-session-info-1"
        )

    # It MUST be status 'done' with the direct answer, NOT 'needs_clarification' offering a procedure!
    assert res["status"] == "done"
    assert "Alarm 13H indicates an inverter overload" in res["answer"]
    assert "Would you like to start the guided procedure?" not in res["answer"]


def test_procedural_query_triggers_procedure_offer():
    """Verify that a procedural query DOES offer a guided procedure when micro-LLM outputs YES."""
    config = {
        "query": {
            "agent": {
                "max_iterations": 2,
                "write_tools": [],
                "clarify_tools": ["request_clarification"]
            }
        },
        "guardrails": {"enabled": False}
    }
    
    mock_store = MockStore()
    
    responses = [
        AIMessage(content="Here are the steps to clean the wheel mounting section:\n1. Loosen bolts.\n2. Wipe surface."),
        AIMessage(content="YES"),
    ]
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = responses
    mock_llm.bind_tools.return_value = mock_llm

    with patch("backend.storage.conversation_store.get_conversation_store", return_value=mock_store), \
         patch("backend.agent.executor._get_primary_citation", return_value=("doc-123", 6)), \
         patch("backend.agent.executor._extract_all_citations", return_value=[{"document_id": "doc-123", "filename": "Manual.pdf", "page": 6}]):
        res = run_agent(
            "how do i clean the wheel mounting section",
            config=config,
            registry={},
            llm=mock_llm,
            session_id="test-session-proc-1"
        )

        assert res["status"] == "needs_clarification"
        assert "Would you like to start the guided procedure?" in res["answer"]
        assert "Continue" in res["options"]


def test_active_procedure_progress_query_answered_from_memory():
    """Verify that asking progress questions during active procedure is answered directly from memory by Procedure Agent."""
    config = {
        "query": {
            "agent": {
                "max_iterations": 2,
                "write_tools": [],
                "clarify_tools": ["request_clarification"]
            }
        },
        "guardrails": {"enabled": False}
    }
    
    mock_store = MockStore()
    mock_store.set_interactive_state("test-session-active-progress", {
        "mode": "guided_assistant",
        "stage": "active",
        "title": "Changeover of Workhead",
        "document_id": "doc-123",
        "filename": "Workhead_Manual.pdf",
        "sections": [
            {"title": "1.1 Changing the Work Spindle Center", "steps": ["(1) Switch to MANU.", "(2) Stop wheel rotation.", "(3) Open front door."]},
            {"title": "1.2 Adjusting Work Spindle Position", "steps": ["(1) Loosen clamp.", "(2) Adjust position."]}
        ],
        "current_sec_idx": 0,
        "current_step_idx": 2, # on Step 3
        "current_idx": 2,
        "steps": ["(1) Switch to MANU.", "(2) Stop wheel rotation.", "(3) Open front door."],
    })

    # Mock Procedure Agent LLM answering progress question directly
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="You have completed 2 steps in Section 1.1 Changing the Work Spindle Center. You are currently on Step 3 of 3: '(3) Open front door.' Section 1.2 is pending."
    )

    with patch("backend.storage.conversation_store.get_conversation_store", return_value=mock_store), \
         patch("backend.retrieval.search_documents.SearchDocumentsTool") as mock_sdt:
        res = run_agent(
            "how many steps are completed?",
            config=config,
            registry={},
            llm=mock_llm,
            session_id="test-session-active-progress"
        )

        # Must NOT call SearchDocumentsTool!
        mock_sdt.assert_not_called()

        assert res["status"] == "needs_clarification"
        assert "You have completed 2 steps in Section 1.1" in res["answer"]
        assert "Still on **Step 3 of 3**" in res["answer"]
        assert "✅ Step Complete - Next" in res["options"]


def test_stuck_technician_offers_global_search_on_refusal():
    """Verify that during an active procedure, asking an out-of-scope question prompts for global search."""
    config = {
        "query": {
            "agent": {
                "max_iterations": 2,
                "write_tools": [],
                "clarify_tools": ["request_clarification"]
            }
        },
        "guardrails": {"enabled": False}
    }
    
    mock_store = MockStore()
    mock_store.set_interactive_state("test-session-subloop", {
        "mode": "guided_assistant",
        "stage": "active",
        "title": "Tailstock Changeover",
        "document_id": "doc-123",
        "filename": "Tailstock_Manual.pdf",
        "sections": [{"title": "Section 1", "steps": ["Step 1", "Step 2"]}],
        "current_sec_idx": 0,
        "current_step_idx": 0,
        "current_idx": 0,
        "steps": ["Step 1", "Step 2"],
    })

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="ACTION:SEARCH")

    mock_sdt = MagicMock()
    mock_sdt.return_value.run.return_value = {
        "answer": "I could not find information about hydraulic pressure in this manual.",
        "sources": []
    }

    with patch("backend.storage.conversation_store.get_conversation_store", return_value=mock_store), \
         patch("backend.retrieval.search_documents.SearchDocumentsTool", mock_sdt):
        res = run_agent(
            "what is the recommended hydraulic pressure?",
            config=config,
            registry={},
            llm=mock_llm,
            session_id="test-session-subloop"
        )

        assert res["status"] == "needs_clarification"
        assert "I could not find information about this in **Tailstock Manual**" in res["answer"]
        assert "Would you like me to search all manuals globally?" in res["answer"]
        assert "✅ Yes, search globally" in res["options"]
        assert "❌ No, that's fine" in res["options"]
