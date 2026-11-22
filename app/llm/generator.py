"""Grounded generation with a real model: the model QUOTES retrieved sentences, code verifies every quote.

The model gets the retrieved policy sentences and must return JSON {"quotes": [...], "answer": "..."}.
  - Every quote must equal one retrieved sentence (whitespace-insensitive); it is replaced by that exact sentence.
  - The model's short `answer` is used only if every sentence of it passes the groundedness guardrail's own support
    test against the retrieved chunks; otherwise the verified quotes are the answer.
  - Anything else (invalid JSON, a quote not in the context, no quotes, a provider error) falls back to the
    capstone's extractive answer from the same chunks, so the text is still retrieved policy text, never invented.
BudgetExceeded is never caught. Fallback text is marked not cacheable, so a transient failure is not stored.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from app.core.cache import is_supported_by
from app.governance.budget import BudgetExceeded, RequestBudget
from app.llm.errors import classify
from app.llm.metering import metered, reported_or_estimated
from app.llm.settings import LLMSettings
from app.rag.documents import split_sentences
from app.rag.evaluation import dedupe
from app.rag.generation import MockGroundedGenerator, _is_complete, contiguous_passages
from app.rag.retrieval import RetrievedChunk

# The JSON answer is two quoted sentences plus a one-sentence answer: well under 400 tokens.
GENERATION_MAX_TOKENS = 400
SYSTEM_PROMPT = (
    "You answer customer questions about Nykaa policies using ONLY the numbered policy sentences you are given. "
    "Return JSON with two fields: \"quotes\": the 1 or 2 sentences that answer the question, copied exactly, "
    "word for word, from the list; \"answer\": a one-sentence answer that uses only facts from those quotes. "
    "Never add facts that are not in the quotes."
)


class GeneratedAnswer(BaseModel):
    quotes: list[str] = Field(min_length=1, max_length=4)
    answer: str = ""


RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "grounded_answer", "schema": GeneratedAnswer.model_json_schema()},
}


class GenerationRejected(ValueError):
    """The model answered, but not in a form that can be verified (the message is the reason)."""


def _norm(text: str) -> str:
    return " ".join(text.split())


class LLMGroundedGenerator(MockGroundedGenerator):
    """A drop-in for MockGroundedGenerator (same `generate(query, chunks)` interface and fallback)."""

    def __init__(self, settings: LLMSettings, http_client: httpx.Client | None = None,
                 max_sentences: int = 2, margin: float = 0.10) -> None:
        super().__init__(max_sentences, margin)
        self.settings = settings
        self.budget: RequestBudget | None = None  # set per request by RagLookupTool
        self.outcomes: list[str] = []  # per call: answer | quotes | fallback:<reason> (measurement evidence)
        self.last_cacheable = True
        self._client = OpenAI(base_url=settings.base_url, api_key=settings.api_key.get_secret_value(),
                              timeout=settings.timeout_s, max_retries=settings.max_retries, http_client=http_client)

    @property
    def identity(self) -> tuple[str, str, str]:
        """Part of the generation cache key: answers from different providers or models are never mixed."""
        return (self.settings.provider, self.settings.base_url, self.settings.model)

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        candidates = dedupe([s for p in contiguous_passages(chunks) for s in split_sentences(p)])
        # Like the mock: fixed-size chunks cut sentences at their edges; offer the model only complete sentences
        # when there are any (measured on Ollama: models otherwise quote fragments such as "specialist, if it...").
        candidates = [s for s in candidates if _is_complete(s)] or candidates
        try:
            text, outcome = self._generate(query, candidates, chunks)
        except BudgetExceeded:
            raise
        except Exception as exc:  # provider down, timeout, bad JSON: degrade to the extractive answer
            error = classify(exc)
            reason = error.kind if error else str(exc) if isinstance(exc, GenerationRejected) else type(exc).__name__
            self.outcomes.append(f"fallback:{reason}")
            self.last_cacheable = False
            return super().generate(query, chunks)
        self.outcomes.append(outcome)
        self.last_cacheable = True
        return text

    def _generate(self, query: str, candidates: list[str], chunks: list[RetrievedChunk]) -> tuple[str, str]:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(candidates, 1))
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Policy sentences:\n{numbered}\n\nCustomer question: {query}"}]
        prompt = SYSTEM_PROMPT + messages[1]["content"]
        with metered(self.budget, prompt, GENERATION_MAX_TOKENS) as spend:
            response = self._client.chat.completions.create(  # type: ignore[call-overload]  # plain dicts
                model=self.settings.model, messages=messages,
                temperature=self.settings.temperature, max_tokens=GENERATION_MAX_TOKENS,
                response_format=RESPONSE_FORMAT,
            )
            content = response.choices[0].message.content or ""
            usage = response.usage
            spend.actual = reported_or_estimated(usage.total_tokens if usage else 0, prompt, content)
        try:
            parsed = GeneratedAnswer.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise GenerationRejected("invalid_json") from exc
        by_norm = {_norm(s): s for s in candidates}
        quotes = [by_norm.get(_norm(q)) for q in parsed.quotes]
        if any(q is None for q in quotes):
            raise GenerationRejected("quote_not_in_context")
        verified = dedupe([q for q in quotes if q])[: self.max_sentences]
        answer = parsed.answer.strip()
        if answer and is_supported_by(answer, chunks):
            return answer, "answer"
        return " ".join(verified), "quotes"
