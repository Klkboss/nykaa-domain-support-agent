"""CrewAI's own OpenAI provider class, pointed at any OpenAI-compatible server and metered per call.

CrewAI's executor makes exactly one `call()` per agent step with native function calling (it passes
`available_functions=None` and executes the returned tool calls itself), so wrapping `call`/`acall` meters every
model request. Tool execution (rag_lookup's own generation) is charged by the tool, not here.
Retries: the OpenAI SDK retries each request LLM_MAX_RETRIES times (429/408/409/5xx/connection, with backoff and
Retry-After); CrewAI's BaseLLM also wraps every `call` in its own rate-limit-only retry (3 attempts). A persistent
429 therefore costs up to 3 x (1 + LLM_MAX_RETRIES) requests; each CrewAI attempt is metered separately.
Provider error text (it can quote the provider's response body) never reaches CrewAI's console: the failed-call
event gets a generic message, and every log record (any logger: CrewAI logs it from several places) has an OpenAI
SDK error body ("Error code: 500 - {...}") cut off at creation, keeping only the status code.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
from crewai.llms import retry as crewai_retry
from crewai.llms.providers.openai.completion import OpenAICompletion
from openai import OpenAI
from pydantic import BaseModel, PrivateAttr

from app.governance.budget import RequestBudget
from app.llm.errors import LLMConfigError
from app.llm.metering import metered, reported_or_estimated
from app.llm.settings import LLMSettings

HIDDEN = "provider error details hidden; see the typed error in the response or the JSONL log"
_SDK_ERROR_BODY = re.compile(r"(Error code: \d{3}) - .*", re.S)


@contextmanager
def without_crewai_rate_limit_retry() -> Iterator[None]:
    """Inside, every CrewAI LLM call runs once: CrewAI's own 429 retry (3 attempts, each waiting Retry-After) is
    skipped, so a failover can start at once. CrewAI skips its loop when this flag says a retry is already running.
    shortcut: a private flag of crewai 1.15.33 (pinned); on a CrewAI upgrade, check that
    test_a_429_with_retry_after_fails_over_at_once_and_the_fallback_keeps_its_retries still passes."""
    token = crewai_retry._active_llm_rate_limit_retry.set(True)
    try:
        yield
    finally:
        crewai_retry._active_llm_rate_limit_retry.reset(token)


def redact_provider_text(message: str) -> str:
    return _SDK_ERROR_BODY.sub(rf"\1 ({HIDDEN})", message)


def install_log_redaction() -> None:
    """Wrap the log-record factory once: every record from every logger is redacted when it is created."""
    previous = logging.getLogRecordFactory()
    if getattr(previous, "_provider_redaction", False):
        return

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        message = record.getMessage()
        redacted = redact_provider_text(message)
        if redacted != message:
            record.msg, record.args = redacted, ()
        return record

    factory._provider_redaction = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


class MeteredOpenAICompletion(OpenAICompletion):
    _budget: RequestBudget | None = PrivateAttr(default=None)
    _calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Every request this LLM made: the same shape as MockCrewLLM.calls (CrewRun reads it)."""
        return self._calls

    def _prompt(self, messages: Any, tools: Any) -> str:
        return str(messages) + str(tools or "")

    def _record(self, messages: Any, usage: int) -> None:
        listed = list(messages) if isinstance(messages, list) else [{"role": "user", "content": str(messages)}]
        self._calls.append({"messages": listed, "tokens": usage})

    def _emit_call_failed_event(self, error: str, from_task: Any = None, from_agent: Any = None) -> None:
        super()._emit_call_failed_event(f"LLM call failed ({HIDDEN})", from_task, from_agent)

    def call(self, messages: Any, tools: Any = None, callbacks: Any = None, available_functions: Any = None,
             from_task: Any = None, from_agent: Any = None, response_model: Any = None) -> Any:
        prompt = self._prompt(messages, tools)
        with metered(self._budget, prompt, self.max_tokens or 0) as spend:
            before = self._token_usage["total_tokens"]
            try:
                result = super().call(messages, tools, callbacks, available_functions, from_task, from_agent,
                                      response_model)
            finally:
                self._record(messages, self._token_usage["total_tokens"] - before)
            spend.actual = reported_or_estimated(self._token_usage["total_tokens"] - before, prompt, str(result))
        return result

    async def acall(self, messages: Any, tools: Any = None, callbacks: Any = None, available_functions: Any = None,
                    from_task: Any = None, from_agent: Any = None, response_model: Any = None) -> Any:
        prompt = self._prompt(messages, tools)
        with metered(self._budget, prompt, self.max_tokens or 0) as spend:
            before = self._token_usage["total_tokens"]
            try:
                result = await super().acall(messages, tools, callbacks, available_functions, from_task,
                                             from_agent, response_model)
            finally:
                self._record(messages, self._token_usage["total_tokens"] - before)
            spend.actual = reported_or_estimated(self._token_usage["total_tokens"] - before, prompt, str(result))
        return result


def json_schema_format(model: type[BaseModel]) -> dict[str, Any]:
    """A response_format the SERVER enforces (constrained decoding) while the answer stays a raw string: CrewAI
    parses a pydantic response_format itself and raises inside the agent, before the task guardrail can retry."""
    return {"type": "json_schema", "json_schema": {"name": model.__name__, "schema": model.model_json_schema()}}


def build_crew_llm(settings: LLMSettings, budget: RequestBudget | None = None,
                   http_client: httpx.Client | None = None,
                   response_format: type[BaseModel] | None = None) -> MeteredOpenAICompletion:
    """A real CrewAI LLM. `response_format` makes every answer schema-constrained JSON (the Composer's SupportDraft),
    returned as text for the caller to validate. `http_client` is for tests (an httpx.MockTransport)."""
    if not settings.is_real:
        raise LLMConfigError("build_crew_llm needs a real provider (LLM_PROVIDER=ollama or openai_compatible).")
    install_log_redaction()
    llm = MeteredOpenAICompletion(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key.get_secret_value(),  # explicit: never falls back to OPENAI_API_KEY
        custom_openai=True,  # Chat Completions only; no fallback to OpenAI's Responses API
        temperature=settings.temperature,
        timeout=settings.timeout_s,
        max_retries=settings.max_retries,
        max_tokens=settings.max_tokens,
        reasoning_effort=settings.reasoning_effort,  # None: not sent
        # CrewAI's own response_format field only takes json_object or a pydantic class (which it parses itself);
        # additional_params is merged into the request as-is.
        additional_params={"response_format": json_schema_format(response_format)} if response_format else {},
    )
    if http_client is not None:  # client_params would also reach the async client, which needs an AsyncClient
        llm._client = OpenAI(**llm._get_client_params(), http_client=http_client)
    llm._budget = budget
    return llm
