"""Taxonomy definitions for document categorization.

This module defines the stable labels used by the categorization engine.

The categorizer is expected to output:
- document_type: one of the supported document type strings
- industry: one of the supported industry strings

Routes are NOT decided by the model; they are derived from config.mapping:
  type_to_route[type_label]
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Taxonomy:
    # Document types: these MUST match config.type_to_route keys.
    document_types = (
        "circuit_diagram",
        "cad_drawing",
        "schematic",
        "invoice",
        "financial_statement",
        "purchase_order",
        "contract",
        "policy",
        "research_paper",
        "report",
        "manual",
        "presentation",
    )

    industries = (
        "automotive",
        "pharma",
        "finance",
        "legal",
        "manufacturing",
        "engineering",
    )


TAXONOMY = Taxonomy()

