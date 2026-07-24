"""Agent-callable tool: look at a document page's ACTUAL rendered image.

Why this exists, and why it's separate from get_page_context: get_page_context
returns the page's raw EXTRACTED TEXT (fast, free, usually enough). But text
extraction can lose visual information entirely — which icon sits next to which
table row, the shape/layout of a diagram, a callout that's positioned rather than
labeled. For those cases the agent needs to actually SEE the page, not read a
description of it.

query.agent's own model (NVIDIA nemotron) is not vision-capable, so this can't be
"send the image straight into the agent's context" — instead this tool renders the
page, sends it + the agent's specific question to a vision-capable model in a
single call, and returns a text answer the agent can use like any other tool
result. Real vision-model call, real cost/latency — the system prompt should
steer the agent to try get_page_context first and reach for this only when text
genuinely isn't enough.
"""
from __future__ import annotations

import glob
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_PATH = os.getenv("CONFIG_PATH", "config/global.yaml")


def _resolve_pdf_path(document_id: str) -> str | None:
    """Same resolution order as the /files/{id}/pdf API endpoint: the DB's
    file_path first, falling back to a glob on the upload id prefix (the DB
    column is often unset — see postgres_store.get_document's column list)."""
    from backend.storage.postgres_store import PostgresStore

    store = PostgresStore()
    try:
        doc = store.get_document(document_id)
    finally:
        store.close()
    if not doc:
        return None

    file_path = doc.get("file_path") or ""
    if file_path and os.path.isfile(file_path):
        return file_path

    matches = glob.glob(os.path.join("uploads", f"{document_id}_*.pdf"))
    return matches[0] if matches else None


class ViewPageImageTool:
    """Agent-callable tool: answer a question by looking at a page's real image.

    Conforms to the AgentTool protocol in backend/agent_tools.py.
    """

    name = "view_page_image"
    description = (
        "Look at the ACTUAL rendered image of one page — for when text extraction "
        "may have lost something visual (which icon lines up with which table row, "
        "a diagram's layout, a callout's position) that get_page_context's plain "
        "text can't convey. Pass document_id, page, and a SPECIFIC question about "
        "what you need to see. This is a real vision-model call — try "
        "get_page_context first; only reach for this when text genuinely isn't "
        "enough to answer confidently."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "document_id from a search_documents citation's source.",
            },
            "page": {
                "type": "integer",
                "description": "Page number from that same citation's source.",
            },
            "question": {
                "type": "string",
                "description": "What you need to know from looking at this page — "
                "be specific (e.g. 'which warning icon is next to the Factor 3 row?').",
            },
        },
        "required": ["document_id", "page", "question"],
    }

    def run(self, document_id: str = "", page: int | str | None = None,
            question: str = "") -> dict[str, Any]:
        if not document_id or page is None:
            return {"error": "document_id and page are both required."}
        try:
            page_no = int(page)
        except (TypeError, ValueError):
            return {"error": f"page must be an integer, got {page!r}."}
        question = (question or "").strip()
        if not question:
            return {"error": "question is required — say what you need to see on this page."}

        pdf_path = _resolve_pdf_path(document_id)
        if not pdf_path:
            return {"error": f"Could not find the PDF file for document_id {document_id!r}."}

        try:
            import fitz
            doc = fitz.open(pdf_path)
            if not (1 <= page_no <= len(doc)):
                doc.close()
                return {"error": f"page {page_no} is out of range (document has {len(doc)} pages)."}
            pix = doc[page_no - 1].get_pixmap(dpi=150)
            png_bytes = pix.tobytes("png")
            doc.close()
        except Exception as exc:
            logger.exception("view_page_image: render failed for %s page %s", document_id, page_no)
            return {"error": f"failed to render page: {exc}"}

        try:
            from backend.core.config import load_config
            from backend.core.vision_client import describe_image
            config = load_config(CONFIG_PATH)
            prompt = (
                "Answer this specific question about the page image, precisely and "
                "concisely, quoting any relevant text/labels VERBATIM. If the answer "
                "isn't visible on this page, say so plainly.\n\nQuestion: " + question
            )
            answer = describe_image(png_bytes, prompt, config).strip()
        except Exception as exc:
            logger.exception("view_page_image: vision call failed for %s page %s", document_id, page_no)
            return {"error": f"vision call failed: {exc}"}

        return {"document_id": document_id, "page": page_no, "question": question, "answer": answer}

    __call__ = run
