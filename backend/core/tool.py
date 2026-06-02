"""The Tool contract and the shared PipelineState.

A tool does ONE job. It READS what it needs from the state and WRITES its
result back. Tools never call each other directly — they only touch the state.
That is what makes the pipeline pluggable.
"""
from __future__ import annotations
from typing import Protocol, TypedDict


class PipelineState(TypedDict, total=False):
    document_id: str
    file_path: str
    file_type: str        # "pdf" | "excel" | "ppt" | "image"
    route: str            # set by categorize; drives the graph
    page_profiles: list   # PageProfile[]
    blocks: list          # NormalizedBlock[]
    chunks: list          # Chunk[]
    errors: list          # append problems here; never raise to kill the run


class Tool(Protocol):
    name: str
    def run(self, state: PipelineState, config: dict) -> PipelineState: ...
