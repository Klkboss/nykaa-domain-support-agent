"""AutoGen's OpenAIChatCompletionClient, pointed at any OpenAI-compatible server and metered per call.

AutoGen re-raises exceptions from inside a team run as a plain RuntimeError string, so the wrapper records the
classified provider error in `.error` for the caller (the same pattern as the budget meter's `exhausted` flag).
Streaming is refused: it would bypass the metering.
Structured output (the review's editor) is read tolerantly: the schema still goes to the server, but the reply is
parsed here, not by the OpenAI SDK, whose parse error crashed the whole review on Cloudflare (measured: E08, and two
gpt-oss-20b demo questions). Strict JSON first, then the first embedded JSON object that fits the schema (fenced or
wrapped in prose), else the caller's `fallback` value; the result is recorded in `.repairs`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any, Literal

import httpx
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
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel, ValidationError

from app.core.mock_llm import json_objects
from app.governance.budget import RequestBudget
from app.llm.crew_llm import json_schema_format
from app.llm.errors import LLMConfigError, LLMError, classify
from app.llm.metering import metered, reported_or_estimated
from app.llm.settings import LLMSettings

# Declared, not detected: the review needs JSON / structured output (the editor returns a Verdict). Function
# calling is declared OFF on purpose: reviewers hold no tools (least autonomy), and AutoGen then refuses any.
MODEL_INFO = ModelInfo(vision=False, function_calling=False, json_output=True, family="unknown",
                       structured_output=True)


class MeteredChatCompletionClient(ChatCompletionClient):
    def __init__(self, inner: ChatCompletionClient, budget: RequestBudget | None, max_tokens: int,
                 fallback: BaseModel | None = None) -> None:
        self.inner = inner
        self.budget = budget
        self.max_tokens = max_tokens
        self.fallback = fallback  # structured output that cannot be read at all becomes this value
        self.error: LLMError | None = None
        self.repairs: list[str] = []  # per structured reply: "extracted" (JSON inside text) or "fallback"

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
        prompt = "".join(str(m.content) for m in messages)
        schema = json_output if isinstance(json_output, type) and issubclass(json_output, BaseModel) else None
        if schema is not None:  # the server still gets the schema; the reply is parsed below, tolerantly
            extra_create_args = {**extra_create_args, "response_format": json_schema_format(schema)}
            json_output = None
        with metered(self.budget, prompt, self.max_tokens) as spend:
            try:
                result = await self.inner.create(messages, tools=tools, tool_choice=tool_choice,
                                                 json_output=json_output, extra_create_args=extra_create_args,
                                                 cancellation_token=cancellation_token)
            except Exception as exc:
                self.error = classify(exc)
                raise
            spend.actual = reported_or_estimated(result.usage.prompt_tokens + result.usage.completion_tokens,
                                                 prompt, str(result.content))
        if schema is not None and isinstance(result.content, str):
            result = result.model_copy(update={"content": self._readable(schema, result.content)})
        return result

    def _readable(self, schema: type[BaseModel], text: str) -> str:
        """Valid JSON for `schema` from a model's reply, or the fallback; raises only if there is no fallback."""
        try:
            return schema.model_validate_json(text).model_dump_json()
        except ValidationError:
            pass
        for obj in json_objects(text):
            try:
                value = schema.model_validate(obj).model_dump_json()
            except ValidationError:
                continue
            self.repairs.append("extracted")
            return value
        if self.fallback is None or not isinstance(self.fallback, schema):
            raise ValueError(f"the reply does not match {schema.__name__}")  # fixed text: never the reply itself
        self.repairs.append("fallback")
        return json.dumps(self.fallback.model_dump())

    def create_stream(self, messages: Sequence[LLMMessage], **kwargs: Any) -> AsyncGenerator[str | CreateResult, None]:
        raise NotImplementedError("streaming is not metered; the review uses create()")

    async def close(self) -> None:
        await self.inner.close()

    def actual_usage(self) -> RequestUsage:
        return self.inner.actual_usage()

    def total_usage(self) -> RequestUsage:
        return self.inner.total_usage()

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self.inner.count_tokens(messages, tools=tools)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self.inner.remaining_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.inner.capabilities

    @property
    def model_info(self) -> ModelInfo:
        return self.inner.model_info


def build_review_client(settings: LLMSettings, budget: RequestBudget | None = None,
                        http_client: httpx.AsyncClient | None = None,
                        fallback: BaseModel | None = None) -> MeteredChatCompletionClient:
    """A real AutoGen model client. `http_client` is for tests (an httpx.MockTransport); `fallback` is the value an
    unreadable structured reply becomes (see the module docstring)."""
    if not settings.is_real:
        raise LLMConfigError("build_review_client needs a real provider (LLM_PROVIDER=ollama or openai_compatible).")
    extra: dict[str, Any] = {"http_client": http_client} if http_client is not None else {}
    if settings.reasoning_effort:
        extra["reasoning_effort"] = settings.reasoning_effort
    inner = OpenAIChatCompletionClient(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key.get_secret_value(),  # explicit: never falls back to OPENAI_API_KEY
        model_info=MODEL_INFO,
        temperature=settings.temperature,
        timeout=settings.timeout_s,
        max_retries=settings.max_retries,
        max_tokens=settings.max_tokens,
        **extra,
    )
    return MeteredChatCompletionClient(inner, budget, settings.max_tokens, fallback)
