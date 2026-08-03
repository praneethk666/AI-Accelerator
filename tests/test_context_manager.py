"""tests/test_context_manager.py

Unit tests for deterministic active context management and ambiguity detection.
"""
import unittest
import time

from backend.agent.context_manager import (
    ActiveContext,
    get_context,
    detect_explicit_entity,
    process_query_context,
    update_context_from_metadata,
)
from backend.guardrails.ambiguity_detector import check_ambiguity


class TestActiveContextManager(unittest.TestCase):

    def setUp(self):
        self.session_id = "test_session_123"
        ctx = get_context(self.session_id)
        ctx.entity_name = None
        ctx.updated_turn = 0

    def test_explicit_entity_detection(self):
        self.assertEqual(detect_explicit_entity("Which component performs reranking?"), "Reranker")
        self.assertEqual(detect_explicit_entity("tell me about embedding model"), "Dense Embedding")
        self.assertEqual(detect_explicit_entity("what is the vision model"), "Vision Model")
        self.assertIsNone(detect_explicit_entity("model name"))

    def test_context_update_from_metadata(self):
        chunks = [
            {
                "content": "The BAAI/bge-reranker-large is used for cross-encoder reranking.",
                "metadata": {"document_id": "doc1.pdf"}
            }
        ]
        update_context_from_metadata(self.session_id, chunks, current_turn=1)
        ctx = get_context(self.session_id)
        self.assertEqual(ctx.entity_name, "Reranker")
        self.assertEqual(ctx.source_document, "doc1.pdf")

    def test_selective_query_rewriting(self):
        # 1. Update context to Reranker at turn 1
        chunks = [{"content": "rerank cross-encoder", "metadata": {}}]
        update_context_from_metadata(self.session_id, chunks, current_turn=1)

        # 2. Ambiguous query "model name" at turn 2 -> rewritten to "Reranker model name"
        rewritten, was_rewritten = process_query_context(self.session_id, "model name", current_turn=2)
        self.assertTrue(was_rewritten)
        self.assertEqual(rewritten, "Reranker model name")

        # 3. Explicit query "tell me about embedding model" at turn 3 -> explicit override to Dense Embedding
        rewritten, was_rewritten = process_query_context(self.session_id, "tell me about embedding model", current_turn=3)
        self.assertFalse(was_rewritten)
        ctx = get_context(self.session_id)
        self.assertEqual(ctx.entity_name, "Dense Embedding")

    def test_context_expiration(self):
        ctx = ActiveContext(entity_name="Reranker", updated_turn=1, updated_timestamp=time.time())
        self.assertTrue(ctx.is_valid(current_turn=2))
        self.assertTrue(ctx.is_valid(current_turn=4))  # 3 turns elapsed
        self.assertFalse(ctx.is_valid(current_turn=5)) # Expired after > 3 turns

    def test_score_delta_ambiguity_detection(self):
        # Tied scores within 0.12 delta -> Ambiguous
        tied_chunks = [
            {"content": "bge-reranker cross-encoder", "rerank_score": 0.85},
            {"content": "nomic-embed dense embedding model", "rerank_score": 0.82},
        ]
        is_ambiguous, options = check_ambiguity(tied_chunks, max_delta=0.12)
        self.assertTrue(is_ambiguous)
        self.assertIn("Reranker Model", options)
        self.assertIn("Dense Embedding Model", options)

        # Clear winner (delta >= 0.12) -> Not ambiguous
        clear_chunks = [
            {"content": "bge-reranker cross-encoder", "rerank_score": 0.95},
            {"content": "nomic-embed dense embedding model", "rerank_score": 0.30},
        ]
        is_ambiguous, options = check_ambiguity(clear_chunks, max_delta=0.12)
        self.assertFalse(is_ambiguous)


if __name__ == "__main__":
    unittest.main()
