"""Structured output (spec Task 9): the schema every crew response must conform to, and the pipeline response.

`SupportDraft` is the Composer task's `output_pydantic` (crewai 1.15.33's crew-level equivalent of
`response_format`). The customer-facing reply is rendered in code from a validated, guardrail-checked draft.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.rag.generation import FALLBACK_ANSWER

OrderStatus = Literal["Placed", "Shipped", "Delivered", "Returned", "Refunded"]
DocId = Annotated[str, StringConstraints(pattern=r"^KB-\d{2,}$")]


class PolicyClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=600)
    source_doc_ids: list[DocId] = Field(min_length=1, max_length=5)


class OrderFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(pattern=r"^ORD-[0-9]{4}$")
    status: OrderStatus
    order_value_inr: int = Field(ge=0)
    escalation_score: float = Field(ge=0.0, le=1.0)
    escalation_threshold: float = Field(ge=0.0, le=1.0)
    escalation_recommended: bool

    @model_validator(mode="after")
    def _decision_matches_threshold(self) -> "OrderFacts":
        if self.escalation_recommended != (self.escalation_score >= self.escalation_threshold):
            raise ValueError("escalation_recommended contradicts escalation_score vs escalation_threshold")
        return self


class SupportDraft(BaseModel):
    """The Response Composer's output. Every crew response must validate against this."""

    model_config = ConfigDict(extra="forbid")

    policy_claims: list[PolicyClaim] = Field(default_factory=list, max_length=5)
    policy_error: str | None = Field(default=None, max_length=200)
    order: OrderFacts | None = None
    order_error: str | None = Field(default=None, max_length=200)
    needs_order_id: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> "SupportDraft":
        if self.order and self.order_error:
            raise ValueError("a draft cannot have both order facts and an order error")
        if self.needs_order_id and (self.order or self.order_error):
            raise ValueError("needs_order_id is only valid when no order was looked up")
        if self.policy_claims and self.policy_error:
            raise ValueError("a draft cannot have both policy claims and a policy error")
        return self


def render_reply(draft: SupportDraft) -> str:
    """The only place customer-facing text is produced: from validated fields, never from free LLM text."""
    parts: list[str] = []
    if draft.policy_error:
        parts.append(f"Policy information: {draft.policy_error}")
    elif draft.policy_claims:
        sources = {tuple(c.source_doc_ids) for c in draft.policy_claims}
        if len(sources) == 1:
            parts.append(f"Policy information (source: {', '.join(sources.pop())}): "
                         + " ".join(c.text for c in draft.policy_claims))
        else:
            parts.append("Policy information: " + " ".join(
                f"{c.text} [{', '.join(c.source_doc_ids)}]" for c in draft.policy_claims))
    if draft.order_error:
        parts.append(f"Order information: {draft.order_error.rstrip('.')}. No order details have been assumed.")
    elif draft.order:
        o = draft.order
        decision = ("meets the escalation threshold, so this case is flagged for escalation" if o.escalation_recommended
                    else "is below the escalation threshold, so no escalation is needed")
        parts.append(f"Order information ({o.record_id}): status {o.status}, order value ₹{o.order_value_inr:,}. "
                     f"Escalation score {o.escalation_score} (threshold {o.escalation_threshold}) {decision}.")
    elif draft.needs_order_id:
        parts.append("Please share your order ID (format ORD-1234) so I can check it.")
    return "\n".join(parts) or FALLBACK_ANSWER


ResponseStatus = Literal["answered", "fallback", "needs_order_id", "blocked", "error", "budget_exceeded"]


class GuardrailReport(BaseModel):
    pii_masked: dict[str, int] = Field(default_factory=dict)
    injection_detected: bool = False
    injection_categories: list[str] = Field(default_factory=list)
    schema_valid: bool | None = None
    grounded: bool | None = None
    rejected_claims: list[str] = Field(default_factory=list)
    removed_citations: list[str] = Field(default_factory=list)
    rejected_order: bool = False
    refused: bool = False  # True when the reply is the "insufficient information" refusal


class SupportResponse(BaseModel):
    session_id: str
    status: ResponseStatus
    reply: str
    masked_question: str
    draft: SupportDraft | None = None
    guardrails: GuardrailReport
