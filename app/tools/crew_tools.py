"""CrewAI tools (spec Task 7): `rag_lookup` for the Retrieval Agent, `check_order_status` for the Lookup Agent.

Both return compact JSON strings, so the next agent receives structured facts. Failures become
{"error": ...} with a generic message and never invented data. Each tool records its own invocations
as execution evidence.
"""

from __future__ import annotations

import json
from typing import Any

from chromadb.api.models.Collection import Collection
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from crewai.tools import BaseTool

from app.core.cache import CachedGenerator
from app.governance.budget import BudgetExceeded, estimate_tokens
from app.guardrails.groundedness import attribute_claims, evidence_by_doc
from app.rag.documents import load_knowledge_base, split_sentences
from app.rag.generation import RECOMMENDED_STRATEGY, SIMILARITY_THRESHOLDS, MockGroundedGenerator, answer_query
from app.rag.vector_store import get_collection, index_documents, persistent_client
from app.tools.order_lookup import ESCALATION_THRESHOLD, OrderLookupError, check_order_status, normalise_record_id


class RagLookupInput(BaseModel):
    query: str = Field(..., description="The customer's policy question, in full.")


class OrderLookupInput(BaseModel):
    record_id: str = Field(..., description="The order id, format ORD-1234.")


def default_policy_collection() -> Collection:
    """The deployed (recommended) collection from chroma_db/, indexed on first use if empty."""
    client = persistent_client()
    collection = get_collection(client, RECOMMENDED_STRATEGY)
    if collection.count() == 0:
        index_documents(client, load_knowledge_base(), RECOMMENDED_STRATEGY)
    return collection


class _MeteredGenerator(MockGroundedGenerator):
    """The grounded-generation model call, charged to the request budget like any other model call (Phase 8)."""

    def __init__(self, inner: MockGroundedGenerator, budget: Any) -> None:
        super().__init__(inner.max_sentences, inner.margin)
        self._inner, self._budget = inner, budget

    def generate(self, query, chunks):
        self._budget.charge(estimate_tokens(query + "".join(c.text for c in chunks)))  # prompt, before the call
        text = self._inner.generate(query, chunks)  # the tool's own generator keeps its call counter
        self._budget.charge(estimate_tokens(text))  # completion, before it is returned
        return text


class RagLookupTool(BaseTool):
    # Named exactly like the spec's pitfall example: "lookup" in the name must not confuse dispatch.
    name: str = "rag_lookup"
    description: str = (
        "Search the Nykaa policy knowledge base and return a grounded answer with its sources and the "
        "retrieved policy text. Returns grounded=false with a fallback when the policies do not cover it."
    )
    args_schema: type[BaseModel] = RagLookupInput
    model_config = ConfigDict(arbitrary_types_allowed=True)

    collection: Any = Field(default=None, exclude=True)
    threshold: float = SIMILARITY_THRESHOLDS[RECOMMENDED_STRATEGY]
    generator: MockGroundedGenerator = Field(default_factory=MockGroundedGenerator, exclude=True)
    budget: Any = Field(default=None, exclude=True)  # app.governance.budget.RequestBudget (Phase 8), or None
    cache: Any = Field(default=None, exclude=True)  # app.core.cache.GenerationCache (Task 16), or None
    _invocations: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _outputs: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _cache_events: list[str] = PrivateAttr(default_factory=list)

    @property
    def invocations(self) -> list[dict[str, Any]]:
        return self._invocations

    @property
    def outputs(self) -> list[dict[str, Any]]:
        """What the tool really returned: the groundedness guardrail's evidence source."""
        return self._outputs

    @property
    def cache_events(self) -> list[str]:
        """Per call: hit / miss / skipped / rejected / error, or not_generated (nothing above the threshold)."""
        return self._cache_events

    def _run(self, query: str) -> str:
        self._invocations.append({"query": query})
        try:
            if self.collection is None:
                self.collection = default_policy_collection()
            generator = self.generator if self.budget is None else _MeteredGenerator(self.generator, self.budget)
            if self.cache is not None:  # outermost: a hit skips the model call and therefore its budget charge
                generator = CachedGenerator(generator, self.cache, self.generator)
            a = answer_query(query, self.collection, self.threshold, generator=generator)
            if self.cache is not None:
                self._cache_events.append(generator.event or "not_generated")
            supporting = [c for c in a.chunks if c.similarity >= a.threshold]
            evidence = evidence_by_doc(supporting)
            result = {
                "tool": self.name,
                "grounded": a.grounded,
                "answer": a.answer,
                "sources": a.sources,  # every doc above the threshold (Phase 3 semantics)
                # Per-sentence citations: only the docs whose retrieved text supports that sentence.
                "claims": attribute_claims(split_sentences(a.answer), evidence) if a.grounded else [],
                "top_similarity": round(a.top_similarity, 4),
                "threshold": a.threshold,
                "evidence": [{"doc_id": d, "passages": p} for d, p in evidence.items()],
                "context": [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "similarity": round(c.similarity, 4),
                             "text": c.text} for c in supporting],
            }
        except BudgetExceeded:
            raise  # a budget stop must stop the request, not look like an unavailable policy search
        except Exception:  # tool boundary: report unavailability, never partial or invented policy text
            result = {"tool": self.name, "error": "Policy search is temporarily unavailable."}
        self._outputs.append(result)
        return json.dumps(result)


class CheckOrderStatusTool(BaseTool):
    name: str = "check_order_status"
    description: str = (
        "Look up one order by its record id and return its status, order value in INR, escalation score, "
        "the escalation threshold and whether escalation is recommended."
    )
    args_schema: type[BaseModel] = OrderLookupInput
    _invocations: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _outputs: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    @property
    def invocations(self) -> list[dict[str, Any]]:
        return self._invocations

    @property
    def outputs(self) -> list[dict[str, Any]]:
        return self._outputs

    def _run(self, record_id: str) -> str:
        self._invocations.append({"record_id": record_id})
        try:
            result = {"tool": self.name, "record_id": normalise_record_id(record_id), **check_order_status(record_id),
                      "escalation_threshold": ESCALATION_THRESHOLD}
        except OrderLookupError as e:  # messages are designed never to echo raw input
            result = {"tool": self.name, "error": str(e)}
        except Exception:
            result = {"tool": self.name, "error": "Order lookup is temporarily unavailable."}
        self._outputs.append(result)
        return json.dumps(result)
