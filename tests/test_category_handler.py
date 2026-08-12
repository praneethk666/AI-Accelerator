from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, ToolMessage
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


class _MockSearchTool:
    name = "search_documents"
    description = "Search documents."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    def __init__(self, result_json: str):
        self.result_json = result_json
        
    def run(self, **kwargs):
        return self.result_json


class MockStore:
    def __init__(self):
        self.state = {}
    def get_interactive_state(self, session_id):
        return self.state.get(session_id)
    def set_interactive_state(self, session_id, state):
        self.state[session_id] = state


_CONFIG = {"query": {"agent": {"max_iterations": 5, "write_tools": []}}}


def test_category_1_process_based():
    # Category 1: Setup procedural query -> expect overview + confirmation pills
    search_res = json.dumps({
        "answer": "Follow these steps: Step 1: Open workpiece holder. Step 2: Set spacing.",
        "citations": [
            {
                "document_id": "doc-1",
                "filename": "20230831_99Y_05_G0797V10_Changeover_workpiece_holder.pdf",
                "page": 2
            }
        ]
    })
    search = _MockSearchTool(search_res)
    mock_store = MockStore()
    
    llm = _ScriptedLLM([
        AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "How to setup workpiece holder?"}, "id": "call_1", "type": "tool_call"}]),
        # Final answer containing steps
        AIMessage(content="To clean or setup, follow these steps:\nStep 1: Open workpiece holder.\nStep 2: Set spacing."),
        # Step extraction prompt call
        AIMessage(content='["Step 1: Open workpiece holder", "Step 2: Set spacing"]')
    ])
    
    mock_blocks = [
        {"text": "Setup steps. Step 1: Open workpiece holder. Step 2: Set spacing.", "source_ref": {"page": 2}}
    ]
    
    with patch("backend.storage.conversation_store.get_conversation_store", return_value=mock_store), \
         patch("backend.storage.postgres_store.PostgresStore") as mock_pg_cls:
        
        mock_pg = MagicMock()
        mock_pg.get_blocks.return_value = mock_blocks
        mock_pg_cls.return_value = mock_pg
        
        result = run_agent(
            "How to setup workpiece holder?",
            config=_CONFIG,
            registry={"search_documents": search},
            llm=llm,
            session_id="cat1_session"
        )
        
    assert result["status"] == "needs_clarification"
    assert "Changeover Workpiece Holder" in result["answer"]
    assert "start the process?" in result["answer"]
    assert result["options"] == ["🚀 Start Guided Process", "No, thanks"]


def test_category_2_cad_drawing():
    # Category 2: CAD schematic/callout query -> returns detailed callout codes and dimensions
    search_res = json.dumps({
        "answer": "Item A08 is part number PSFJ10-86-VC7 from MISUMI, Quantity: 1.",
        "citations": [
            {
                "document_id": "cad-doc-1",
                "filename": "spindle_assembly_drawing.pdf",
                "page": 1
            }
        ]
    })
    search = _MockSearchTool(search_res)
    
    llm = _ScriptedLLM([
        AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "What is the part no for A08 callout in spindle assembly?"}, "id": "call_2", "type": "tool_call"}]),
        AIMessage(content="Item A08 is part number PSFJ10-86-VC7 from MISUMI, Quantity: 1 [spindle_assembly_drawing, p.1].")
    ])
    
    result = run_agent(
        "What is the part no for A08 callout in spindle assembly?",
        config=_CONFIG,
        registry={"search_documents": search},
        llm=llm,
        session_id="cat2_session"
    )
    
    assert result["status"] == "done"
    assert "PSFJ10-86-VC7" in result["answer"]
    assert "MISUMI" in result["answer"]


def test_category_3_direct_fact():
    # Category 3: Direct fact/drawing number QA -> direct answer, 1 turn, no checklists offered
    search_res = json.dumps({
        "answer": "The drawing number for spindle assembly is KE-MC000954-G.",
        "citations": [
            {
                "document_id": "doc-3",
                "filename": "machine_manual.pdf",
                "page": 5
            }
        ]
    })
    search = _MockSearchTool(search_res)
    
    llm = _ScriptedLLM([
        AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "What is the drawing no for spindle assembly?"}, "id": "call_3", "type": "tool_call"}]),
        AIMessage(content="The drawing number for spindle assembly is KE-MC000954-G [machine_manual, p.5].")
    ])
    
    result = run_agent(
        "What is the drawing no for spindle assembly?",
        config=_CONFIG,
        registry={"search_documents": search},
        llm=llm,
        session_id="cat3_session"
    )
    
    assert result["status"] == "done"
    assert "KE-MC000954-G" in result["answer"]
    assert "Guided Process" not in result["answer"]
