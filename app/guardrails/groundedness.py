"""Output guardrail (spec Task 10): refuse what the retrieved policy context does not support.

Evidence is taken from the ACTUAL tool outputs recorded during the crew run, never from what an agent says
the tools returned.
- A policy claim is kept only for the cited documents whose retrieved evidence supports it: the claim appears
  verbatim, or every number in it appears and at least MIN_COVERAGE of its content words do. Citations that do
  not support the claim are removed; a claim with no supporting citation is rejected.
- Order facts must equal a real check_order_status result. An order result is never policy evidence.
- If nothing supported remains, the reply falls back to the refusal sentence (render_reply).
"""

from __future__ import annotations

import re
from typing import Any

from app.rag.generation import contiguous_passages
from app.rag.retrieval import RetrievedChunk
from app.schemas import GuardrailReport, OrderFacts, PolicyClaim, SupportDraft

MIN_COVERAGE = 0.8  # paraphrase allowance; numbers must match exactly
_WORD_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_STOPWORDS = {"the", "and", "for", "with", "your", "you", "are", "our", "can", "will", "this", "that", "from",
              "within", "has", "have", "its", "their", "they", "was", "were", "not", "any", "all", "only"}


def content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _normalise(text: str) -> str:
    return " ".join(text.split())


def claim_supported(claim: str, evidence: list[str]) -> bool:
    joined = _normalise(" ".join(evidence))
    if _normalise(claim) in joined:
        return True
    if not set(_NUMBER_RE.findall(claim)) <= set(_NUMBER_RE.findall(joined)):
        return False  # a changed number (days, ₹, months) is never a paraphrase
    words = content_words(claim)
    return bool(words) and len(words & content_words(joined)) / len(words) >= MIN_COVERAGE


def evidence_by_doc(chunks: list[RetrievedChunk]) -> dict[str, list[str]]:
    """Per-document retrieved text; overlapping fixed windows are rejoined so a sentence is not split."""
    by_doc: dict[str, list[RetrievedChunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    return {doc: contiguous_passages(cs) for doc, cs in by_doc.items()}


def attribute_claims(sentences: list[str], evidence: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Cite each sentence only with the documents whose evidence actually supports it."""
    return [{"text": s, "source_doc_ids": [d for d, ev in evidence.items() if claim_supported(s, ev)]}
            for s in sentences]


def check_draft(draft: SupportDraft, rag_output: dict | None, order_output: dict | None) -> tuple[SupportDraft, GuardrailReport]:
    report = GuardrailReport(schema_valid=True)
    evidence: dict[str, list[str]] = {}
    if rag_output and rag_output.get("grounded"):
        evidence = {e["doc_id"]: e["passages"] for e in rag_output.get("evidence", [])}

    kept: list[PolicyClaim] = []
    for claim in draft.policy_claims:
        supporting = [d for d in claim.source_doc_ids if d in evidence and claim_supported(claim.text, evidence[d])]
        report.removed_citations += [f"{d}: {claim.text[:80]}" for d in claim.source_doc_ids if d not in supporting]
        if supporting:
            kept.append(PolicyClaim(text=claim.text, source_doc_ids=supporting))
        else:
            report.rejected_claims.append(claim.text)

    order = draft.order
    if order is not None:
        real = OrderFacts.model_validate({k: order_output[k] for k in OrderFacts.model_fields}) if (
            order_output and "error" not in order_output) else None
        if real != order:
            report.rejected_order = True
            order = None

    order_error = draft.order_error
    if order_error is not None and not (order_output and order_output.get("error") == order_error):
        report.rejected_order = True  # an error the tool never reported is also an invented order fact
        order_error = None

    checked = SupportDraft(policy_claims=kept, policy_error=draft.policy_error, order=order,
                           order_error=order_error, needs_order_id=draft.needs_order_id)
    report.grounded = not report.rejected_claims and not report.rejected_order
    return checked, report
