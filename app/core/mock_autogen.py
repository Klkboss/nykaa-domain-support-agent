"""Deterministic AutoGen ChatCompletionClient used under MOCK_LLM.

CrewAI's BaseLLM is not an AutoGen model client, so AutoGen gets its own adapter that
implements autogen_core.models.ChatCompletionClient. One client per agent role:
- "reviewer": lists draft sentences not supported by the retrieved context.
- "editor":   turns the reviewer's findings into the structured verdict (json_output model).
The task message carries the draft and the context in "DRAFT:" / "CONTEXT:" sections.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    ModelInfo,
    RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema

from app.guardrails.groundedness import claim_supported
from app.rag.documents import split_sentences

UNSUPPORTED_PREFIX = "UNSUPPORTED: "
ALL_SUPPORTED = "ALL CLAIMS SUPPORTED"


def is_supported(sentence: str, context: str) -> bool:
    """Same rule as the output groundedness guardrail (verbatim, or all numbers + 80% content words)."""
    return claim_supported(sentence, [context])


def parse_sections(task: str) -> tuple[str, str]:
    """Split a review task into (draft, context)."""
    match = re.search(r"DRAFT:\s*(.*?)\s*CONTEXT:\s*(.*)", task, re.S)
    if not match:
        raise ValueError("review task must contain DRAFT: and CONTEXT: sections")
    return match.group(1).strip(), match.group(2).strip()


class MockChatCompletionClient(ChatCompletionClient):
    def __init__(self, role: Literal["reviewer", "editor"], reviewer_name: str = "Policy_Compliance_Reviewer",
                 budget: Any = None) -> None:
        self.role = role
        self.reviewer_name = reviewer_name
        self.budget = budget  # app.governance.budget.RequestBudget, or None (unmetered)
        self.calls: list[dict[str, Any]] = []
        self._total = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._last = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        prompt_tokens = self.count_tokens(messages)
        if self.budget is not None:
            self.budget.charge(prompt_tokens)  # before the prompt is "sent"
        task = next(str(m.content) for m in messages if getattr(m, "source", None) == "user")
        draft, context = parse_sections(task)
        if self.role == "reviewer":
            content = self._review(draft, context)
        else:
            # Only the reviewer's generated turn is read, never the task text.
            review = next((str(m.content) for m in messages if getattr(m, "source", None) == self.reviewer_name), "")
            content = self._edit(draft, review, json_output)
        if self.budget is not None:
            self.budget.charge(len(content) // 4)  # before the completion is returned
        self._last = RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=len(content) // 4)
        self._total = RequestUsage(
            prompt_tokens=self._total.prompt_tokens + self._last.prompt_tokens,
            completion_tokens=self._total.completion_tokens + self._last.completion_tokens,
        )
        self.calls.append({"json_output": json_output, "content": content})
        return CreateResult(finish_reason="stop", content=content, usage=self._last, cached=False)

    @staticmethod
    def _review(draft: str, context: str) -> str:
        flagged = [s for s in split_sentences(draft) if not is_supported(s, context)]
        return "\n".join(UNSUPPORTED_PREFIX + s for s in flagged) if flagged else ALL_SUPPORTED

    @staticmethod
    def _edit(draft: str, review: str, json_output: bool | type[BaseModel] | None) -> str:
        flagged = [line[len(UNSUPPORTED_PREFIX):] for line in review.splitlines() if line.startswith(UNSUPPORTED_PREFIX)]
        if flagged:
            kept = [s for s in split_sentences(draft) if s not in flagged]
            verdict = {
                "approved": False,
                "final_answer": " ".join(kept) or "I'm sorry, I can't confirm that from our policy documents.",
                "reason": f"Removed {len(flagged)} unsupported claim(s): " + " | ".join(flagged),
            }
        else:
            verdict = {"approved": True, "final_answer": draft, "reason": "All claims are supported by the retrieved context."}
        if isinstance(json_output, type) and issubclass(json_output, BaseModel):
            return json_output.model_validate(verdict).model_dump_json()
        return json.dumps(verdict)

    async def create_stream(self, messages: Sequence[LLMMessage], **kwargs: Any) -> AsyncGenerator[str | CreateResult, None]:
        yield await self.create(messages, **kwargs)

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._last

    def total_usage(self) -> RequestUsage:
        return self._total

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return sum(len(str(m.content)) for m in messages) // 4  # documented estimate: ~4 chars/token

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 128_000 - self.count_tokens(messages)

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore[override]
        return ModelCapabilities(vision=False, function_calling=False, json_output=True)

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(vision=False, function_calling=False, json_output=True, family="unknown", structured_output=True)
