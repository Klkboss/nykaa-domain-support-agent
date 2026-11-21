"""CrewAI's own OpenAI provider class, pointed at any OpenAI-compatible server and metered per call.

CrewAI's executor makes exactly one `call()` per agent step with native function calling (it passes
`available_functions=None` and executes the returned tool calls itself), so wrapping `call`/`acall` meters every
model request. Tool execution (rag_lookup's own generation) is charged by the tool, not here.
Retries: the OpenAI SDK retries each request LLM_MAX_RETRIES times (429/408/409/5xx/connection, with backoff and
Retry-After); CrewAI's BaseLLM also wraps every `call` in its own rate-limit-only retry (3 attempts). A persistent
429 therefore costs up to 3 x (1 + LLM_MAX_RETRIES) requests; each CrewAI attempt is metered separately.
"""

from __future__ import annotations

from typing import Any

import httpx
from crewai.llms.providers.openai.completion import OpenAICompletion
from openai import OpenAI
from pydantic import PrivateAttr

from app.governance.budget import RequestBudget
from app.llm.errors import LLMConfigError
from app.llm.metering import metered, reported_or_estimated
from app.llm.settings import LLMSettings


class MeteredOpenAICompletion(OpenAICompletion):
    _budget: RequestBudget | None = PrivateAttr(default=None)

    def _prompt(self, messages: Any, tools: Any) -> str:
        return str(messages) + str(tools or "")

    def call(self, messages: Any, tools: Any = None, callbacks: Any = None, available_functions: Any = None,
             from_task: Any = None, from_agent: Any = None, response_model: Any = None) -> Any:
        prompt = self._prompt(messages, tools)
        with metered(self._budget, prompt, self.max_tokens or 0) as spend:
            before = self._token_usage["total_tokens"]
            result = super().call(messages, tools, callbacks, available_functions, from_task, from_agent,
                                  response_model)
            spend.actual = reported_or_estimated(self._token_usage["total_tokens"] - before, prompt, str(result))
        return result

    async def acall(self, messages: Any, tools: Any = None, callbacks: Any = None, available_functions: Any = None,
                    from_task: Any = None, from_agent: Any = None, response_model: Any = None) -> Any:
        prompt = self._prompt(messages, tools)
        with metered(self._budget, prompt, self.max_tokens or 0) as spend:
            before = self._token_usage["total_tokens"]
            result = await super().acall(messages, tools, callbacks, available_functions, from_task, from_agent,
                                         response_model)
            spend.actual = reported_or_estimated(self._token_usage["total_tokens"] - before, prompt, str(result))
        return result


def build_crew_llm(settings: LLMSettings, budget: RequestBudget | None = None,
                   http_client: httpx.Client | None = None) -> MeteredOpenAICompletion:
    """A real CrewAI LLM. `http_client` is for tests (an httpx.MockTransport); production uses the SDK default."""
    if not settings.is_real:
        raise LLMConfigError("build_crew_llm needs a real provider (LLM_PROVIDER=ollama or openai_compatible).")
    llm = MeteredOpenAICompletion(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key.get_secret_value(),  # explicit: never falls back to OPENAI_API_KEY
        custom_openai=True,  # Chat Completions only; no fallback to OpenAI's Responses API
        temperature=settings.temperature,
        timeout=settings.timeout_s,
        max_retries=settings.max_retries,
        max_tokens=settings.max_tokens,
    )
    if http_client is not None:  # client_params would also reach the async client, which needs an AsyncClient
        llm._client = OpenAI(**llm._get_client_params(), http_client=http_client)
    llm._budget = budget
    return llm
