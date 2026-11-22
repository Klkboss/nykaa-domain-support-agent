"""Pydantic request/response models for the FastAPI layer (spec Task 11)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.rag.documents import Category
from app.schemas import SupportResponse

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN,
                                   description="Omit (or use an unknown id) to start a fresh session.")


class AskResponse(SupportResponse):
    trace_id: str
    new_session: bool


class AddDocumentRequest(BaseModel):
    """Same fields as the KB's KBDocument; the text must be 2-5 sentences."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=r"^KB-\d{2,}$")
    title: str = Field(min_length=3, max_length=120)
    topic: str = Field(pattern=r"^[a-z][a-z_]*$", max_length=60)
    categories: list[Category] = Field(min_length=1)
    text: str = Field(min_length=1, max_length=2000)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    chunks_indexed: dict[str, int]
    persistent: Literal[False] = False  # per-process index, rebuilt from the KB at startup


class WsQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)


class ErrorBody(BaseModel):
    type: Literal["validation_error", "conflict", "internal_error", "rate_limited"]
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    trace_id: str
    error: ErrorBody


class Explanation(BaseModel):
    """v2 chat UI, opt-in (/ws/chat?explain=1): how one reply was made, from that request's own evidence."""

    cited_sources: list[str]  # KB ids the reply cites
    retrieved_sources: list[str]  # KB ids rag_lookup returned above the similarity threshold
    tools: list[str]  # tool calls in order (the Retrieval Agent, then the Lookup Agent)
    grounded: bool | None  # the groundedness guardrail's verdict (None: no crew ran, e.g. a blocked question)
    rejected_claims: int
    review: Literal["approved", "revised", "not_run"]
    escalation_recommended: bool | None  # None: no order was looked up
    injection_categories: list[str]
    provider: str  # mock, ollama, or the hosted provider's host (e.g. api.groq.com): who actually answered
    model: str
    failover: bool = False  # v2: the primary provider failed and the fallback (provider above) was used
    latency_ms: float
    tokens: int  # charged to this request's token budget (estimates for the mock, reported usage for real models)
