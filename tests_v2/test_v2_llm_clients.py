"""v2 Stage A: the metered CrewAI and AutoGen clients against a fake OpenAI-compatible server (mocked HTTP).

Covers: request shape, real tool calling, budget reserve-then-settle on reported usage, retries with backoff,
typed errors (timeout, unreachable, rate limit, auth, unknown model, other status), and keys never leaking.
"""

from __future__ import annotations

import asyncio

import httpx
import openai
import pytest
from autogen_core.models import UserMessage
from crewai.llms.retry import _LLM_RATE_LIMIT_MAX_ATTEMPTS
from pydantic import BaseModel

from app.governance.budget import BudgetExceeded, RequestBudget, estimate_tokens
from app.llm.autogen_client import build_review_client
from app.llm.crew_llm import build_crew_llm
from app.llm.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMModelNotFound,
    LLMRateLimited,
    LLMResponseError,
    LLMTimeout,
    LLMUnavailable,
    classify,
)
from app.llm.settings import LLMSettings, load_llm_settings
from tests_v2.fake_openai import BASE_URL, FakeServer, completion, status

MESSAGES = [{"role": "user", "content": "How long do I have to return shoes?"}]
TOOLS = [{"type": "function", "function": {
    "name": "check_order_status", "description": "Look up one order.",
    "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}, "required": ["record_id"]}}}]


def settings(**overrides) -> LLMSettings:
    base = {"provider": "openai_compatible", "model": "fake-model", "base_url": BASE_URL, "max_retries": 0,
            "max_tokens": 256}
    return LLMSettings.model_validate(base | overrides)


def crew_call(server: FakeServer, budget: RequestBudget | None = None, tools=None, **overrides):
    llm = build_crew_llm(settings(**overrides), budget, http_client=server.sync_client())
    return llm.call(MESSAGES, tools=tools)


# --- CrewAI LLM ---------------------------------------------------------------------------------------------

def test_crew_llm_sends_an_openai_compatible_request_and_returns_text():
    server = FakeServer(completion("Within 30 days."))
    assert crew_call(server) == "Within 30 days."
    (body,) = server.bodies()
    assert body["model"] == "fake-model" and body["temperature"] == 0.0 and body["max_tokens"] == 256
    assert body["messages"][-1]["content"] == MESSAGES[0]["content"]
    assert str(server.requests[0].url) == f"{BASE_URL}/chat/completions"


def test_crew_llm_returns_native_tool_calls_for_the_executor():
    call = {"id": "c1", "type": "function", "function": {"name": "check_order_status",
                                                          "arguments": '{"record_id": "ORD-1016"}'}}
    server = FakeServer(completion(None, tool_calls=[call]))
    result = crew_call(server, tools=TOOLS)
    assert isinstance(result, list) and result[0].function.name == "check_order_status"
    assert server.bodies()[0]["tools"][0]["function"]["name"] == "check_order_status"


def test_budget_is_charged_the_reported_usage():
    budget = RequestBudget(5_000)
    crew_call(FakeServer(completion(usage=(120, 30))), budget)
    assert budget.used == 150


def test_missing_usage_is_charged_as_the_estimate():
    budget = RequestBudget(5_000)
    result = crew_call(FakeServer(completion("x" * 400, usage=None)), budget)
    assert budget.used == estimate_tokens(str(MESSAGES) + "") + estimate_tokens(result) > 0


def test_a_call_that_cannot_fit_is_refused_before_anything_is_sent():
    server = FakeServer(completion())
    with pytest.raises(BudgetExceeded):
        crew_call(server, RequestBudget(100))  # reservation = prompt estimate + 256 > 100
    assert server.requests == []


def test_usage_beyond_the_reservation_is_charged_and_can_stop_the_request():
    budget = RequestBudget(300)
    with pytest.raises(BudgetExceeded):
        crew_call(FakeServer(completion(usage=(250, 100))), budget)  # reserved ~266, reported 350 > cap 300
    assert budget.exhausted


def test_rate_limit_is_retried_with_backoff_then_succeeds():
    server = FakeServer(status(429, **{"retry-after-ms": "1"}), completion("ok", usage=(5, 1)))
    budget = RequestBudget(5_000)
    assert crew_call(server, budget, max_retries=2) == "ok"
    assert len(server.requests) == 2 and budget.used == 6


@pytest.mark.parametrize("step, expected", [
    (status(429, **{"retry-after-ms": "1"}), LLMRateLimited),
    (httpx.ReadTimeout("slow"), LLMTimeout),
    (httpx.ConnectError("refused"), LLMUnavailable),
    (status(401), LLMAuthError),
    (status(404), LLMModelNotFound),
    (status(500), LLMResponseError),
])
def test_provider_failures_become_typed_errors_and_release_the_reservation(step, expected):
    budget = RequestBudget(5_000)
    server = FakeServer(step)
    with pytest.raises(Exception) as err:
        crew_call(server, budget, max_retries=1)
    error = classify(err.value)
    assert isinstance(error, expected)
    assert "fake provider error" not in str(error)  # the provider's body is never echoed
    assert budget.used == 0
    sdk_attempts = 2 if expected in (LLMRateLimited, LLMTimeout, LLMUnavailable, LLMResponseError) else 1
    # CrewAI's BaseLLM adds its own rate-limit-only retry (3 attempts) around each metered call
    crewai_attempts = _LLM_RATE_LIMIT_MAX_ATTEMPTS if expected is LLMRateLimited else 1
    assert len(server.requests) == sdk_attempts * crewai_attempts


def test_only_the_configured_key_is_sent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-leak")
    server = FakeServer(completion())
    llm = build_crew_llm(load_llm_settings({"LLM_PROVIDER": "ollama"}, mock=False), None, server.sync_client())
    llm.call(MESSAGES)
    assert server.requests[0].headers["authorization"] == "Bearer not-set"
    keyed = FakeServer(completion())
    build_crew_llm(settings(api_key="sk-configured"), None, keyed.sync_client()).call(MESSAGES)
    assert keyed.requests[0].headers["authorization"] == "Bearer sk-configured"


def test_mock_settings_cannot_build_a_real_client():
    with pytest.raises(LLMConfigError):
        build_crew_llm(LLMSettings())
    with pytest.raises(LLMConfigError):
        build_review_client(LLMSettings())


def test_classify_ignores_unrelated_errors():
    assert classify(ValueError("x")) is None
    assert classify(RuntimeError("wrapped")) is None
    try:
        try:
            raise httpx.ConnectError("refused")
        except httpx.ConnectError as inner:
            raise RuntimeError("agent failed") from inner
    except RuntimeError as outer:
        assert isinstance(classify(outer), LLMUnavailable)


# --- AutoGen client -----------------------------------------------------------------------------------------

class Verdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


def review_create(server: FakeServer, budget: RequestBudget | None = None, json_output=None):
    client = build_review_client(settings(), budget, http_client=server.async_client())
    result = asyncio.run(client.create([UserMessage(content="DRAFT: x CONTEXT: y", source="user")],
                                       json_output=json_output))
    return client, result


def test_review_client_returns_structured_output_and_charges_reported_usage():
    verdict = '{"approved": true, "final_answer": "x", "reason": "supported"}'
    server = FakeServer(completion(verdict, usage=(40, 12)))
    budget = RequestBudget(5_000)
    _, result = review_create(server, budget, json_output=Verdict)
    assert Verdict.model_validate_json(result.content).approved is True
    assert budget.used == 52
    body = server.bodies()[0]
    assert body["model"] == "fake-model" and body["response_format"]["type"] == "json_schema"


def test_review_client_records_the_typed_error_and_releases_the_reservation():
    budget = RequestBudget(5_000)
    server = FakeServer(status(404))
    client = build_review_client(settings(), budget, http_client=server.async_client())
    with pytest.raises(openai.NotFoundError):
        asyncio.run(client.create([UserMessage(content="hi", source="user")]))
    assert isinstance(client.error, LLMModelNotFound) and budget.used == 0


def test_review_client_refuses_before_sending_when_the_budget_cannot_fit():
    server = FakeServer(completion())
    client = build_review_client(settings(), RequestBudget(10), http_client=server.async_client())
    with pytest.raises(BudgetExceeded):
        asyncio.run(client.create([UserMessage(content="hi", source="user")]))
    assert server.requests == []


def test_review_client_refuses_unmetered_streaming():
    client = build_review_client(settings(), None, http_client=FakeServer(completion()).async_client())
    with pytest.raises(NotImplementedError):
        client.create_stream([UserMessage(content="hi", source="user")])
