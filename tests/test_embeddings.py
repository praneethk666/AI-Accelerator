"""Embedder unit tests (no infra).  Run: python tests/test_embeddings.py (or pytest)"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.embeddings.local_embedder import DEFAULT_DIM, LocalEmbedder, embed_text


def test_deterministic_and_dim():
    assert embed_text("power rail") == embed_text("power rail")  # reproducible
    assert len(embed_text("x")) == DEFAULT_DIM
    assert len(LocalEmbedder(8).embed("x")) == 8  # dim is configurable


def test_distinct_text_distinct_vector():
    assert embed_text("op-amp") != embed_text("resistor")


def test_unit_length():
    v = embed_text("5V rail")
    assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-6  # normalized


if __name__ == "__main__":
    test_deterministic_and_dim()
    test_distinct_text_distinct_vector()
    test_unit_length()
    print("embedding tests passed")
