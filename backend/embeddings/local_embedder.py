"""Local deterministic embedder.

- stub: hashes text -> a fixed-dim unit vector (NOT semantic — plumbing only)
- deterministic: same text -> same vector, so tests are reproducible offline
- no external call; a real model swaps in later via config (embeddings.model)
"""

from __future__ import annotations

import hashlib
import math

DEFAULT_DIM = 256


def embed_text(text: str, dim: int = DEFAULT_DIM) -> list[float]:
    """Map text to a deterministic unit-length vector of length `dim`."""
    out: list[float] = []
    counter = 0
    while len(out) < dim:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        for i in range(0, len(digest), 4):
            if len(out) >= dim:
                break
            n = int.from_bytes(digest[i : i + 4], "big") / 2**32  # [0, 1)
            out.append(n * 2 - 1)  # [-1, 1)
        counter += 1
    norm = math.sqrt(sum(x * x for x in out)) or 1.0  # normalize -> cosine-friendly
    return [x / norm for x in out]


class LocalEmbedder:
    """Embedder with a fixed dimension; the unit the embed tool wraps."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        return embed_text(text, self.dim)
