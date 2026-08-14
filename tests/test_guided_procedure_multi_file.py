from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from backend.agent.executor import run_agent


class _ScriptedLLM:
    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.invocations: list[list] = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invocations.append(list(messages))
        if self._responses:
            return self._responses.pop(0)
        return AIMessage(content="Fallback response")


class _MultiFileSearchTool:
    name = "search_documents"
    description = "Search documents."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    def run(self, **kwargs):
        return json.dumps({
            "answer": "Here is how to clean the wheel mounting or coolant tank.",
            "citations": [
                {
                    "document_id": "doc-1",
                    "filename": "20230831_99Y_03_G0738V30_Cleaning_up_the_wheel_mounting_section.pdf",
                    "page": 1
                },
                {
                    "document_id": "doc-2",
                    "filename": "20230831_99Y_22_Coolant_tank_CNK.pdf",
                    "page": 1
                }
            ]
        })


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


_CONFIG = {"query": {"agent": {"max_iterations": 5, "write_tools": []}}}


def test_guided_procedure_flow():
    search = _MultiFileSearchTool()
    mock_store = MockStore()
    
    # 1. Test 1 (Disambiguation): Send query "how to clean the machine"
    # The agent will search documents and receive multi-file citations, triggering the menu.
    llm1 = _ScriptedLLM([
        AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "how to clean the machine"}, "id": "call_1", "type": "tool_call"}]),
        AIMessage(content="Here is cleaning details."),
    ])
    
    with patch("backend.storage.conversation_store.get_conversation_store", return_value=mock_store):
        result = run_agent(
            "how to clean the machine",
            config=_CONFIG,
            registry={"search_documents": search},
            llm=llm1,
            session_id="test_session_1"
        )
        
    assert result["status"] == "needs_clarification"
    assert "different manuals" in result["answer"]
    assert "1. Cleaning Up The Wheel Mounting Section" in result["answer"]
    assert "2. Coolant Tank Cnk" in result["answer"]
    assert result["options"] == ["1. Cleaning Up The Wheel Mounting Section", "2. Coolant Tank Cnk"]
    
    # Verify state was saved
    saved_state = mock_store.get_interactive_state("test_session_1")
    assert saved_state["mode"] == "guided_assistant"
    assert saved_state["stage"] == "disambiguation"
    assert len(saved_state["disambiguation_options"]) == 2

    # 2. Test 2 (Overview & Confirmation): Send choice "1"
    # The handler will intercept, load selected document content, run overview LLM, and offer confirmation.
    llm2 = _ScriptedLLM([
        AIMessage(content="This is an overview of the wheel mounting cleaning procedure."), # Mock overview invoke
        AIMessage(content='["Step 1: Turn off power", "Step 2: Clean the wheel flange"]'), # Mock steps extraction invoke
    ])
    
    # Mock PostgresStore get_blocks to return mock document blocks
    mock_blocks = [
        {"text": "Wheel Mounting Section Details. Step 1: Turn off power. Step 2: Clean the wheel flange.", "source_ref": {"page": 1}}
    ]
    
    with patch("backend.storage.conversation_store.get_conversation_store", return_value=mock_store), \
         patch("backend.storage.postgres_store.PostgresStore") as mock_pg_cls:
        
        mock_pg = MagicMock()
        mock_pg.get_blocks.return_value = mock_blocks
        mock_pg_cls.return_value = mock_pg
        
        result2 = run_agent(
            "1. Cleaning Up The Wheel Mounting Section",
            config=_CONFIG,
            registry={},
            llm=llm2,
            session_id="test_session_1"
        )
        
    assert result2["status"] == "needs_clarification"
    assert "Overview of Cleaning Up The Wheel Mounting Section" in result2["answer"]
    assert "Shall we start?" in result2["answer"]
    assert result2["options"] == ["🚀 Start Guided Process", "No, thanks"]
    
    saved_state = mock_store.get_interactive_state("test_session_1")
    assert saved_state["stage"] == "overview"
    assert saved_state["steps"] == ["Step 1: Turn off power", "Step 2: Clean the wheel flange"]

    # 3. Test 3 (Step 1 Only): Send "🚀 Start Guided Process"
    # The handler should transition to active and present Step 1 with LOTO warning.
    with patch("backend.storage.conversation_store.get_conversation_store", return_value=mock_store), \
         patch("backend.storage.postgres_store.PostgresStore") as mock_pg_cls:
        
        mock_pg = MagicMock()
        mock_pg.list_documents.return_value = [] # No CAD drawings
        mock_pg_cls.return_value = mock_pg
        
        result3 = run_agent(
            "🚀 Start Guided Process",
            config=_CONFIG,
            registry={},
            llm=_ScriptedLLM([]),
            session_id="test_session_1"
        )
        
    assert result3["status"] == "needs_clarification"
    assert "SAFETY MANDATE" in result3["answer"]
    assert "Step 1 of 2" in result3["answer"]
    assert "Step 1: Turn off power" in result3["answer"]
    assert "📋 View Full Section Summary" in result3["options"]
    
    saved_state = mock_store.get_interactive_state("test_session_1")
    assert saved_state["stage"] == "active"
    assert saved_state["current_idx"] == 0

    # 3.5. Test 3.5 (Stuck Technician Troubleshooting Loop): Send question "Where is the breaker?"
    # The agent should answer it using context, and remain on Step 1.
    llm_trouble = _ScriptedLLM([
        AIMessage(content="The main power breaker is located on the back panel of the machine enclosure.")
    ])
    with patch("backend.storage.conversation_store.get_conversation_store", return_value=mock_store), \
         patch("backend.storage.postgres_store.PostgresStore") as mock_pg_cls:
         
        mock_pg = MagicMock()
        mock_pg.get_blocks.return_value = [{"text": "Main power breaker is on back panel.", "source_ref": {"page": 1}}]
        mock_pg_cls.return_value = mock_pg
        
        result_trouble = run_agent(
            "Where is the breaker?",
            config=_CONFIG,
            registry={},
            llm=llm_trouble,
            session_id="test_session_1"
        )
        
    assert result_trouble["status"] == "needs_clarification"
    assert "back panel" in result_trouble["answer"]
    assert "Step 1 of 2" in result_trouble["answer"]
    assert "📋 View Full Section Summary" in result_trouble["options"]
    
    # Verify we stayed on Step 1
    saved_state = mock_store.get_interactive_state("test_session_1")
    assert saved_state["current_idx"] == 0

    # 4. Test 4 (Step 2 & Summary): Send "✅ Step Complete - Next"
    # It should show Step 2, then after next Done it should generate Case Summary.
    with patch("backend.storage.conversation_store.get_conversation_store", return_value=mock_store):
        result4 = run_agent(
            "✅ Step Complete - Next",
            config=_CONFIG,
            registry={},
            llm=_ScriptedLLM([]),
            session_id="test_session_1"
        )
        
    assert result4["status"] == "needs_clarification"
    assert "Step 2 of 2" in result4["answer"]
    assert "Step 2: Clean the wheel flange" in result4["answer"]
    
    saved_state = mock_store.get_interactive_state("test_session_1")
    assert saved_state["current_idx"] == 1
    
    # Confirming Step 2 should complete and generate Celebration Message
    llm_summary = _ScriptedLLM([
        AIMessage(content="🎉 All steps are completely finished! Status: Resolved. Do you need anything else?")
    ])
    with patch("backend.storage.conversation_store.get_conversation_store", return_value=mock_store):
        result5 = run_agent(
            "✅ Step Complete - Next",
            config=_CONFIG,
            registry={},
            llm=llm_summary,
            session_id="test_session_1"
        )
        
    assert result5["status"] == "done"
    assert "All steps are completely finished!" in result5["answer"]
    assert "Status: Resolved" in result5["answer"]
    
    # State should be cleared
    assert mock_store.get_interactive_state("test_session_1") is None
