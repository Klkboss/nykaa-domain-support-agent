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
    type: Literal["validation_error", "conflict", "internal_error"]
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    trace_id: str
    error: ErrorBody
