from __future__ import annotations

from unittest.mock import patch

from backend.retrieval.search_documents import search_documents


def test_search_documents_wraps_run_query_and_formats_sources():
    final_state = {
        "answer": "The M6 bolt torque is 12 Nm.",
        "citations": [
            {
                "document_id": "doc-fixture-001",
                "filename": "sample.pdf",
                "page": 1,
                "sheet": None,
                "slide": None,
                "snippet": "The assembly requires an M6 bolt torqued to 12 Nm.",
                "summary": "Torque specification for the M6 assembly bolt.",
                "image_path": None,
                "table_data": None,
            }
        ],
    }
    config = {"query": {"steps": ["query_planner", "retrieval", "answerer"]}}
    registry = object()

    with patch(
        "backend.retrieval.search_documents._run_query",
        return_value=final_state,
    ) as mock_run_query:
        result = search_documents(
            "What is the torque specification for bolt M6?",
            ["doc-fixture-001"],
            registry=registry,
            config=config,
        )

    mock_run_query.assert_called_once_with(
        "What is the torque specification for bolt M6?",
        registry,
        config,
        session_id="",
        document_scope=["doc-fixture-001"],
        doc_type=None,
        industry=None,
        conversation_history=None,
    )
    assert result == {
        "answer": "The M6 bolt torque is 12 Nm.",
        "citations": final_state["citations"],
        "trace_id": None,
        "sources": [
            {
                "document_id": "doc-fixture-001",
                "filename": "sample.pdf",
                "page": 1,
                "score": None,
                "sheet": None,
                "slide": None,
                "summary": "Torque specification for the M6 assembly bolt.",
                "snippet": "The assembly requires an M6 bolt torqued to 12 Nm.",
                "image_path": None,
            }
        ],
    }

def test_build_sources_deduplication_keeps_different_slides_and_sheets():
    from backend.retrieval.search_documents import _build_sources
    citations = [
        # Slide 4
        {
            "document_id": "doc-1",
            "filename": "deck.pptx",
            "page": None,
            "slide": 4,
            "sheet": None,
            "score": 0.9,
            "snippet": "slide 4 snippet",
        },
        # Slide 5 (same doc, different slide)
        {
            "document_id": "doc-1",
            "filename": "deck.pptx",
            "page": None,
            "slide": 5,
            "sheet": None,
            "score": 0.8,
            "snippet": "slide 5 snippet",
        },
        # Slide 4 duplicate (same doc, same slide) - should be deduplicated
        {
            "document_id": "doc-1",
            "filename": "deck.pptx",
            "page": None,
            "slide": 4,
            "sheet": None,
            "score": 0.7,
            "snippet": "another slide 4 snippet",
        },
    ]
    sources = _build_sources(citations)
    assert len(sources) == 2
    assert sources[0]["slide"] == 4
    assert sources[1]["slide"] == 5
