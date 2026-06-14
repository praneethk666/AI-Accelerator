"""The Tool contract and the shared PipelineState.

A tool does ONE job. It READS what it needs from the state and WRITES its
result back. Tools never call each other directly — they only touch the state.
That is what makes the pipeline pluggable.
"""
from __future__ import annotations
from typing import Protocol, TypedDict


class PipelineState(TypedDict, total=False):
    # ── ingestion ──────────────────────────────────────────────────
    document_id: str
    file_path: str
    file_type: str           # "pdf" | "excel" | "ppt" | "image"
    pdf_kind: str            # "digital" | "scanned" | "mixed"; set by categorize (detector) — picks the PDF extractor
    document_type: str       # set by categorize
    industry: str            # set by categorize
    confidence: float        # categorize confidence score
    reasoning: str           # categorize: why it picked this type/route (debug/observability)
    route: str               # set by categorize; drives the graph
    page_profiles: list      # PageProfile[]
    blocks: list             # NormalizedBlock[]
    chunks: list             # Chunk[]
    errors: list             # append problems here; never raise to kill the run

    # ── query time ─────────────────────────────────────────────────
    query: str               # raw user question
    session_id: str          # conversation session
    document_scope: list     # document_ids to search; empty = whole collection
    conversation_history: list  # loaded by run_query before the graph starts
    standalone_query: str    # contextualised query (after follow-up rewrite)
    sub_questions: list      # decomposed sub-questions for retrieval
    skip_retrieval: bool     # set by adaptive_router when corpus fits in context
    retrieval_retry: bool    # set by retrieval_judge to trigger one re-retrieval
    retrieved_chunks: list   # Chunk[] returned by retrieval / rerank
    answer: str              # final answer text
    citations: list          # citation dicts (filename, page, snippet, image_path, table_data)


class Tool(Protocol):
    name: str
    def run(self, state: PipelineState, config: dict) -> PipelineState: ...
