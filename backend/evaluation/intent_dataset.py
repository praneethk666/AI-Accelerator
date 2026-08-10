"""Labelled cases for agent intent classification.

Data only — no LLM, no network, no heavy imports — so the dataset can be linted,
unit-tested, and extended without running an evaluation.

Each case is a message plus the intent it SHOULD receive. The label decides one
thing: whether the agent is forced to search the document corpus on turn 0
(document_question / action) or may answer directly (follow_up / general).

Adding cases: keep every label defensible. A case whose "right" answer is genuinely
arguable makes the accuracy number less meaningful, not more rigorous — if it is a
real judgement call, write why in `note` so a future reader can re-litigate it.
Follow-ups MUST carry history: without a prior turn there is nothing to follow up on.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.agent.intent_classifier import (
    ACTION,
    DOCUMENT_QUESTION,
    FOLLOW_UP,
    GENERAL,
)

# A realistic prior exchange, reused by the follow-up cases.
BRAKE_HISTORY: tuple[tuple[str, str], ...] = (
    ("user", "what does the service manual say about front brake torque?"),
    ("assistant", "The manual specifies 42 Nm for the front brake caliper bolts, "
                  "tightened in a star pattern (Service Manual, p.87)."),
)

INVOICE_HISTORY: tuple[tuple[str, str], ...] = (
    ("user", "what's the total on invoice 4471?"),
    ("assistant", "Invoice 4471 totals $18,420.00, due net-30 (invoice_4471.pdf, p.1)."),
)


@dataclass(frozen=True)
class Case:
    """One labelled message.

    message:  what the user types.
    expected: the intent it should be assigned.
    history:  prior turns as (role, text); required for follow-ups.
    note:     why this label, when the case is a boundary or judgement call.
    """

    message: str
    expected: str
    history: tuple[tuple[str, str], ...] = ()
    note: str = ""


# ── document_question — needs the ingested corpus ────────────────────────────
_DOCUMENT_QUESTIONS = [
    Case("what is the torque spec for the front brake?", DOCUMENT_QUESTION),
    Case("what does the warranty cover?", DOCUMENT_QUESTION),
    Case("how many units were sold in Q3?", DOCUMENT_QUESTION),
    Case("what is the part number for the oil filter?", DOCUMENT_QUESTION),
    Case("summarise the safety section of the manual", DOCUMENT_QUESTION),
    Case("what are the recommended maintenance intervals?", DOCUMENT_QUESTION),
    Case("who signed the contract?", DOCUMENT_QUESTION),
    Case("what is the total on invoice 4471?", DOCUMENT_QUESTION),
    Case("which components are listed in the circuit diagram?", DOCUMENT_QUESTION),
    Case("what voltage does the datasheet specify?", DOCUMENT_QUESTION),
    Case("find the section about lubrication", DOCUMENT_QUESTION),
    Case("what does the report say about revenue growth?", DOCUMENT_QUESTION),
    Case("are there any warnings about the battery?", DOCUMENT_QUESTION,
         note="absence-style question; still has to search to answer it"),
    Case("what is the operating temperature range?", DOCUMENT_QUESTION),
    Case("show me the payment terms", DOCUMENT_QUESTION),
    Case("what's in section 3.2?", DOCUMENT_QUESTION),
    Case("according to the spec sheet, what is the max load?", DOCUMENT_QUESTION),
    Case("does the policy mention remote work?", DOCUMENT_QUESTION),
    Case("compare the Q1 and Q2 figures in the financial statement", DOCUMENT_QUESTION),
    Case("what did the audit find?", DOCUMENT_QUESTION),
    Case("model name", DOCUMENT_QUESTION,
         note="terse corpus lookup — short phrases like this must still search"),
    Case("in manual.pdf, what does it say about coolant?", DOCUMENT_QUESTION,
         note="names a file but asks for its CONTENT, so it is a search not an action"),
    Case("list the safety warnings in the document", DOCUMENT_QUESTION,
         note="'list' phrasing, but the subject is document CONTENT, not the file inventory"),
    Case("what are the dimensions of the housing?", DOCUMENT_QUESTION),
    Case("pull up what the drawing says about tolerances", DOCUMENT_QUESTION),
    Case("how much did we invoice ACME last quarter?", DOCUMENT_QUESTION),
    Case("what certifications does the product hold?", DOCUMENT_QUESTION),
    Case("is there a clause about termination?", DOCUMENT_QUESTION),
    Case("what's the recommended torque sequence?", DOCUMENT_QUESTION),
    Case("give me the key points from the research paper", DOCUMENT_QUESTION),
]

# ── action — perform an operation, not answer from content ───────────────────
_ACTIONS = [
    Case("ingest uploads/report.pdf", ACTION),
    Case("upload this file into the system", ACTION),
    Case("list all the documents", ACTION),
    Case("what files have been ingested?", ACTION),
    Case("show me the loaded documents", ACTION),
    Case("please index uploads/manual.pdf", ACTION),
    Case("run a SQL query against the database", ACTION),
    Case("add sales_2026.xlsx to the corpus", ACTION),
    Case("how many documents are in the system?", ACTION,
         note="counts the file inventory, not content inside the files"),
    Case("re-ingest the manual, it changed", ACTION),
    Case("what's the status of the last upload?", ACTION),
    Case("import C:/data/contract.docx", ACTION),
    Case("process the spreadsheet I just added", ACTION),
    Case("query the database for the top 10 customers", ACTION),
    Case("show me which files failed to process", ACTION),
]

# ── follow_up — answerable from the conversation above ───────────────────────
_FOLLOW_UPS = [
    Case("summarise that", FOLLOW_UP, BRAKE_HISTORY),
    Case("explain your last answer", FOLLOW_UP, BRAKE_HISTORY),
    Case("what did you just say about the torque?", FOLLOW_UP, BRAKE_HISTORY),
    Case("can you say that more simply?", FOLLOW_UP, BRAKE_HISTORY),
    Case("repeat that in bullet points", FOLLOW_UP, BRAKE_HISTORY),
    Case("why did you say 42 Nm?", FOLLOW_UP, BRAKE_HISTORY),
    Case("translate your previous answer to Spanish", FOLLOW_UP, BRAKE_HISTORY),
    Case("shorten that to one sentence", FOLLOW_UP, BRAKE_HISTORY),
    Case("what was my previous question?", FOLLOW_UP, BRAKE_HISTORY),
    Case("expand on what you just told me", FOLLOW_UP, BRAKE_HISTORY),
    Case("put that in a table", FOLLOW_UP, INVOICE_HISTORY),
    Case("what does net-30 mean in what you just said?", FOLLOW_UP, INVOICE_HISTORY,
         note="defines a term FROM the previous answer — no new retrieval needed"),
    Case("say that again", FOLLOW_UP, INVOICE_HISTORY),
    Case("rephrase your answer for a non-engineer", FOLLOW_UP, BRAKE_HISTORY),
    Case("which page did you get that from?", FOLLOW_UP, BRAKE_HISTORY,
         note="the citation is already in the prior answer"),
]

# ── general — own knowledge, reasoning, chit-chat ────────────────────────────
_GENERAL = [
    Case("what is 2+2?", GENERAL),
    Case("what does RAG stand for?", GENERAL),
    Case("who are you and what can you do?", GENERAL),
    Case("convert 42 Nm to foot-pounds", GENERAL),
    Case("what's the difference between a vector and a scalar?", GENERAL),
    Case("thanks, that's helpful", GENERAL,
         note="pleasantry; a model may read it as follow_up, but both take the "
              "direct-answer path so routing is unaffected"),
    Case("can you write a haiku about engineering?", GENERAL),
    Case("what year did the first moon landing happen?", GENERAL),
    Case("explain what an embedding is in general terms", GENERAL),
    Case("how are you doing today?", GENERAL),
    Case("what is the capital of France?", GENERAL),
    Case("what's a reasonable torque wrench to buy?", GENERAL,
         note="shopping advice from world knowledge, not from the user's documents"),
    Case("spell 'accelerator' backwards", GENERAL),
    Case("what does ISO 9001 mean generally?", GENERAL,
         note="general standard definition; not asking what the CORPUS says about it"),
    Case("give me a tip for writing clearer prompts", GENERAL),
]

CASES: list[Case] = [*_DOCUMENT_QUESTIONS, *_ACTIONS, *_FOLLOW_UPS, *_GENERAL]

# Labels whose route forces a document search on turn 0.
TOOL_REQUIRING_LABELS = frozenset({DOCUMENT_QUESTION, ACTION})
ALL_LABELS = [DOCUMENT_QUESTION, ACTION, FOLLOW_UP, GENERAL]


def cases_for(label: str) -> list[Case]:
    """Every case expected to carry `label`."""
    return [c for c in CASES if c.expected == label]


def to_messages(history: tuple[tuple[str, str], ...]) -> list:
    """Convert (role, text) pairs into LangChain messages for the classifier."""
    from langchain_core.messages import AIMessage, HumanMessage

    return [
        HumanMessage(text) if role == "user" else AIMessage(text)
        for role, text in history
    ]
