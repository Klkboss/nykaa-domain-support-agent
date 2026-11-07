"""Spec Task 15: least autonomy (Application layer) and the per-request token budget (Runtime layer)."""

from __future__ import annotations

import json
import uuid

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo
from autogen_core.tools import FunctionTool
from crewai import Agent
from crewai.hooks.dispatch import HookAborted
from fastapi.testclient import TestClient

import app.pipeline as pipeline
from app.agents.crew import build_support_crew, run_support_crew
from app.api.main import create_app
from app.core.cache import GenerationCache
from app.core.config import ROOT
from app.core.mock_autogen import MockChatCompletionClient
from app.core.mock_llm import MockCrewLLM
from app.governance.budget import (MAX_REQUEST_TOKENS, BudgetExceeded, RequestBudget, estimate_tokens,
                                   project_request_tokens)
from app.governance.least_autonomy import (ALLOWED_TOOLS, LeastAutonomyViolation, enforce_crew_least_autonomy,
                                           enforce_reviewers_have_no_tools)
from app.memory.session_memory import get_session_history, has_session
from app.pipeline import BUDGET_REPLY, SupportAssistant, format_history
from app.rag.generation import RECOMMENDED_STRATEGY
from app.review.autogen_review import build_review_team
from app.tools.crew_tools import CheckOrderStatusTool, OrderLookupInput, RagLookupTool

PHONE = "+91 98765 43210"
OVERSIZED = "Please explain every return rule for every category in full detail. " * 300  # ~20,000 characters


def sid() -> str:
    return f"gov-{uuid.uuid4().hex[:8]}"


def assistant(collections, **kw) -> SupportAssistant:
    return SupportAssistant(rag_tool_factory=lambda: RagLookupTool(collection=collections[RECOMMENDED_STRATEGY]), **kw)


# --- Application layer: least autonomy ------------------------------------------------------------

def agent(role: str, tools: list, **kw) -> Agent:
    return Agent(role=role, goal="g", backstory="b", tools=tools, llm=MockCrewLLM(), allow_delegation=False, **kw)


def real_agents() -> dict[str, Agent]:
    return {a.role: a for a in build_support_crew().crew.agents}


def test_real_crew_grants_each_agent_exactly_its_allow_list():
    agents = real_agents()
    assert {r: {t.name for t in a.tools} for r, a in agents.items()} == ALLOWED_TOOLS
    assert [r for r, a in agents.items() if any(isinstance(t, CheckOrderStatusTool) for t in a.tools)] == ["Lookup Agent"]
    assert not any(a.allow_delegation for a in agents.values())
    enforce_crew_least_autonomy(list(agents.values()))  # positive case: no violation


def test_allow_list_still_holds_after_a_real_kickoff_that_used_both_tools(collections):
    support = build_support_crew(RagLookupTool(collection=collections[RECOMMENDED_STRATEGY]))
    support.crew.kickoff(inputs={"question": "My order ORD-1029 arrived damaged. What is your damaged item policy?",
                                 "history": "(none)"})
    assert support.order_tool.invocations == [{"record_id": "ORD-1029"}] and support.rag_tool.invocations
    enforce_crew_least_autonomy(support.crew.agents)  # CrewAI added no tools at run time


class RenamedOrderTool(CheckOrderStatusTool):
    name: str = "order_status"


class DisguisedOrderTool(CheckOrderStatusTool):
    name: str = "rag_lookup"  # an allowed name, but the order-lookup argument schema


@pytest.mark.parametrize("make, message", [
    (lambda: agent("Retrieval Agent", [RagLookupTool(), CheckOrderStatusTool()]), "must hold exactly"),
    (lambda: agent("Response Composer", [CheckOrderStatusTool()]), "must hold exactly"),
    (lambda: agent("Retrieval Agent", [RenamedOrderTool()]), "must hold exactly"),
    (lambda: agent("Retrieval Agent", [DisguisedOrderTool()]), "only the Lookup Agent"),
    (lambda: agent("Lookup Agent", [CheckOrderStatusTool(), RagLookupTool()]), "must hold exactly"),
    (lambda: agent("Lookup Agent", []), "must hold exactly"),
    (lambda: Agent(role="Response Composer", goal="g", backstory="b", tools=[], llm=MockCrewLLM(),
                   allow_delegation=True), "may not delegate"),
    (lambda: agent("Refund Agent", [CheckOrderStatusTool()]), "no tool grant"),
])
def test_any_other_wiring_of_the_order_tool_is_refused(make, message):
    assert DisguisedOrderTool().args_schema is OrderLookupInput
    with pytest.raises(LeastAutonomyViolation, match=message):
        enforce_crew_least_autonomy([make()])


def test_crew_builder_refuses_to_build_a_crew_that_breaks_the_allow_list(monkeypatch):
    import app.agents.crew as crew_module

    real = crew_module.Agent

    def wire_order_tool_to_composer(**kw):
        if kw["role"] == "Response Composer":
            kw["tools"] = [CheckOrderStatusTool()]
        return real(**kw)

    monkeypatch.setattr(crew_module, "Agent", wire_order_tool_to_composer)
    with pytest.raises(LeastAutonomyViolation, match="Response Composer"):
        build_support_crew()


def check_order_status(record_id: str) -> str:  # an AutoGen-style copy of the order tool
    return ""


def test_autogen_reviewers_hold_no_tools():
    team = build_review_team()
    enforce_reviewers_have_no_tools(team._participants)  # positive case
    assert all(not a._tools and not a._handoff_tools for a in team._participants)


def test_autogen_refuses_tools_for_the_reviewers_model_client():
    order_tool = FunctionTool(check_order_status, description="order lookup")
    with pytest.raises(ValueError, match="does not support function calling"):
        AssistantAgent("Policy_Compliance_Reviewer", model_client=MockChatCompletionClient("reviewer"),
                       tools=[order_tool])


def test_guard_refuses_a_reviewer_with_tools_even_on_a_capable_client():
    class CapableClient(MockChatCompletionClient):
        @property
        def model_info(self) -> ModelInfo:
            return ModelInfo(vision=False, function_calling=True, json_output=True, family="unknown",
                             structured_output=True)

    order_tool = FunctionTool(check_order_status, description="order lookup")
    armed = AssistantAgent("Policy_Compliance_Reviewer", model_client=CapableClient("reviewer"), tools=[order_tool])
    with pytest.raises(LeastAutonomyViolation, match="may not hold tools"):
        enforce_reviewers_have_no_tools([armed])


# --- Runtime layer: per-request token budget --------------------------------------------------------

def test_budget_boundaries_below_at_and_above_the_cap():
    b = RequestBudget(cap=100)
    b.charge(99)  # below
    b.charge(1)   # exactly at the cap: allowed
    assert b.used == 100
    with pytest.raises(BudgetExceeded) as e:
        b.charge(1)  # above: refused without spending
    assert b.used == 100 and (e.value.needed, e.value.used, e.value.cap) == (1, 100, 100)
    assert isinstance(e.value, HookAborted)  # CrewAI re-raises HookAborted instead of retrying the task


def test_projection_formula_and_cap():
    assert estimate_tokens("x" * 400) == 100
    assert project_request_tokens("x" * 400, "y" * 400) == 7_000 + 6 * 200
    assert MAX_REQUEST_TOKENS == 12_000
    assert project_request_tokens("x" * 2000) <= MAX_REQUEST_TOKENS  # every question the API accepts fits alone


def test_projection_is_an_upper_bound_of_real_spend(collections):
    a = assistant(collections, max_request_tokens=10**9)
    single = [q["query"] for q in json.loads((ROOT / "evaluation" / "test_queries.json")
                                            .read_text(encoding="utf-8"))["queries"]]
    session = sid()  # plus one multi-turn session that starts with the longest question the API accepts
    multi = [("Please tell me about the return window for my new dress. " * 40)[:2000], "What about shoes?",
             "And for ORD-1016?"]
    for s, q in [(sid(), q) for q in single] + [(session, q) for q in multi]:
        history = format_history(get_session_history(s).messages) if has_session(s) else ""
        r = a.ask(s, q)
        if r.status == "blocked":
            continue
        assert 0 < a.last_budget.used <= project_request_tokens(q, history), (q[:40], a.last_budget.used)


def test_request_below_the_cap_is_answered_and_metered(collections):
    a = assistant(collections)
    r = a.ask(sid(), "How long do I have to return shoes?")
    assert r.status == "answered" and a.last_budget.cap == MAX_REQUEST_TOKENS
    # crew calls (prompt + completion of each) and the review reservation are all charged
    assert a.last_review.reserved_tokens < a.last_budget.used <= MAX_REQUEST_TOKENS


def test_request_projected_exactly_at_the_cap_is_allowed_one_over_is_refused(collections):
    q = "How long do I have to return shoes?"
    at = assistant(collections, max_request_tokens=project_request_tokens(q))
    assert at.ask(sid(), q).status == "answered" and at.last_budget.used <= at.max_request_tokens
    over = assistant(collections, max_request_tokens=project_request_tokens(q) - 1)
    r = over.ask(sid(), q)
    assert r.status == "budget_exceeded" and r.reply == BUDGET_REPLY and over.last_crew_run is None


def test_repeated_requests_each_get_a_fresh_budget(collections):
    # Budget isolation only, so no cache reuse (max_entries=0 keeps nothing): a Task 16 cache hit legitimately
    # skips the generation charge; that effect is tested in tests/test_cache.py.
    a = assistant(collections, generation_cache=GenerationCache(max_entries=0))
    s = sid()
    first = (a.ask(s, "How long do I have to return shoes?"), a.last_budget)
    second = (a.ask(s, "How long do I have to return shoes?"), a.last_budget)
    third = (a.ask(sid(), "How long do I have to return shoes?"), a.last_budget)
    assert all(r.status == "answered" for r, _ in (first, second, third))
    assert first[1] is not second[1] is not third[1]  # one budget per request
    assert first[1].used == third[1].used  # nothing carried over from earlier requests
    assert second[1].used > first[1].used  # the repeat in the same session also pays for its history


def test_oversized_request_is_refused_before_any_model_call_and_never_stored(collections):
    a = assistant(collections)
    s = sid()
    r = a.ask(s, f"{OVERSIZED} Call me on {PHONE}.")
    assert project_request_tokens(r.masked_question) > MAX_REQUEST_TOKENS
    assert r.status == "budget_exceeded" and r.reply == BUDGET_REPLY and r.draft is None
    assert a.last_crew_run is None and a.last_budget is None  # no crew, no model call, nothing spent
    assert not has_session(s)  # not stored: cannot inflate later turns
    assert PHONE not in r.masked_question and "[PHONE]" in r.masked_question  # input guardrails ran first
    assert a.ask(s, "How long do I have to return shoes?").status == "answered"  # the session still works


def test_budget_exhausted_mid_crew_stops_the_request_safely(collections, monkeypatch):
    monkeypatch.setattr(pipeline, "project_request_tokens", lambda q, h="": 0)  # let it past the pre-flight
    a = assistant(collections, max_request_tokens=2_000)
    r = a.ask(sid(), "How long do I have to return shoes?")
    assert r.status == "budget_exceeded" and r.reply == BUDGET_REPLY and r.draft is None
    assert 0 < a.last_budget.used <= 2_000  # spent up to the cap, never beyond


def test_budget_exhausted_at_the_review_stage_stops_the_request_safely(collections, monkeypatch):
    q = "How long do I have to return shoes?"
    probe = assistant(collections)
    probe.ask(sid(), q)
    # everything before the review = total minus the review's metered spend (its unused reservation was released)
    crew_spend, reserve = probe.last_budget.used - probe.last_review.spent_tokens, probe.last_review.reserved_tokens
    monkeypatch.setattr(pipeline, "project_request_tokens", lambda q, h="": 0)
    a = assistant(collections, max_request_tokens=crew_spend + reserve - 1)
    r = a.ask(sid(), q)
    assert r.status == "budget_exceeded" and r.reply == BUDGET_REPLY
    assert a.last_budget.used == crew_spend <= a.max_request_tokens  # the crew ran; the review was not started


def test_crew_stops_at_the_first_call_over_budget_without_retries(collections):
    budget = RequestBudget(cap=500)  # smaller than the first prompt
    with pytest.raises(BudgetExceeded):
        run_support_crew("How long do I have to return shoes?",
                         rag_tool=RagLookupTool(collection=collections[RECOMMENDED_STRATEGY]), budget=budget)
    assert budget.used == 0


# --- API: unchanged for valid sizes, safe when the budget refuses -------------------------------------

@pytest.fixture(scope="module")
def api(tmp_path_factory):
    log = tmp_path_factory.mktemp("govlogs") / "requests.jsonl"
    with TestClient(create_app(log_path=log)) as c:
        yield c, log


def test_api_accepts_the_largest_question_it_allows(api):
    client, _ = api
    question = ("Please tell me about the return window for my new dress. " * 40)[:2000]
    assert len(question) == 2000
    r = client.post("/ask", json={"question": question})
    assert r.status_code == 200 and r.json()["status"] == "answered"


def test_api_reports_a_budget_refusal_safely_and_logs_it_masked(api):
    client, log = api
    client.app.state.assistant.max_request_tokens = 100  # a deliberately tiny cap for this check
    try:
        r = client.post("/ask", json={"question": f"Call me on {PHONE}. How long do I have to return shoes?"})
    finally:
        client.app.state.assistant.max_request_tokens = MAX_REQUEST_TOKENS
    body = r.json()
    assert r.status_code == 200 and body["status"] == "budget_exceeded" and body["reply"] == BUDGET_REPLY
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    entry = next(e for e in entries if e.get("trace_id") == body["trace_id"] and e.get("response_status"))
    assert entry["response_status"] == "budget_exceeded" and "98765" not in log.read_text(encoding="utf-8")


# --- Complete model-call accounting (Phase 8 review) ---------------------------------------------------

import sys  # noqa: E402

from pydantic import PrivateAttr  # noqa: E402

import app.review.autogen_review as autogen_review  # noqa: E402
from app.review.autogen_review import review_policy_draft  # noqa: E402
from app.schemas import PolicyClaim, SupportDraft  # noqa: E402

SHOES = "How long do I have to return shoes?"
FOOTWEAR = "Footwear can be returned within 30 days of delivery if it is unworn and sent back in its original shoebox."


def spy_charges(monkeypatch) -> list[tuple[int, str, int]]:
    """Record (budget id, charging code path, tokens) for every RequestBudget.charge."""
    charges, real = [], RequestBudget.charge

    def spy(self, tokens):
        charges.append((id(self), sys._getframe(1).f_code.co_qualname, tokens))
        return real(self, tokens)

    monkeypatch.setattr(RequestBudget, "charge", spy)
    return charges


def test_every_model_call_in_a_request_is_charged(collections, monkeypatch):
    charges = spy_charges(monkeypatch)
    a = assistant(collections)
    a.ask(sid(), SHOES)
    run, review, budget = a.last_crew_run, a.last_review, a.last_budget
    by = lambda path: [c for c in charges if c[1] == path]  # noqa: E731
    crew_calls = sum(run.llm_calls.values())
    assert len(by("MockCrewLLM.call")) == 2 * crew_calls and crew_calls >= 4  # prompt + completion, every agent call
    assert len(by("_MeteredGenerator.generate")) == 2  # the grounded-generation call inside rag_lookup
    assert len(by("MockChatCompletionClient.create")) == 4  # reviewer + editor, prompt + completion each
    assert len(by("review_policy_draft")) == 1  # the reservation
    meter = by("MockChatCompletionClient.create")[0][0]
    assert {c[0] for c in charges} == {id(budget), meter}  # only the request budget and the review's meter
    assert len(charges) == 2 * crew_calls + 2 + 4 + 1  # nothing else, and nothing uncharged
    request_side = sum(t for b, p, t in charges if b == id(budget) and p != "review_policy_draft")
    review_side = sum(t for _, p, t in charges if p == "MockChatCompletionClient.create")
    assert review_side == review.spent_tokens <= review.reserved_tokens
    assert budget.used == request_side + review_side  # the unused reservation was released


def test_grounded_generation_call_is_charged(collections):
    budget = RequestBudget()
    tool = RagLookupTool(collection=collections[RECOMMENDED_STRATEGY], budget=budget)
    out = json.loads(tool._run(SHOES))
    from app.rag.query_analysis import analyse_query

    assert analyse_query(SHOES).search_text == SHOES  # not routed: the generator sees the question itself
    prompt = SHOES + "".join(c["text"] for c in out["context"])  # question + the chunks above the threshold
    assert out["grounded"] and tool.generator.calls == 1
    assert budget.used == estimate_tokens(prompt) + estimate_tokens(out["answer"])
    with pytest.raises(BudgetExceeded):  # a budget stop is not swallowed as "policy search unavailable"
        RagLookupTool(collection=collections[RECOMMENDED_STRATEGY], budget=RequestBudget(cap=10))._run(SHOES)


class FailsOnce(MockCrewLLM):
    _failed: bool = PrivateAttr(default=False)

    def _respond(self, messages, agent_tools):
        if not self._failed:
            self._failed = True
            raise ValueError("transient model error")
        return super()._respond(messages, agent_tools)


def test_retried_model_calls_are_charged(collections, monkeypatch):
    charges = spy_charges(monkeypatch)
    assistant(collections).ask(sid(), SHOES)
    normal = len([c for c in charges if c[1] == "MockCrewLLM.call"])
    charges.clear()
    a = assistant(collections, composer_llm_factory=FailsOnce)
    r = a.ask(sid(), SHOES)
    retried = len([c for c in charges if c[1] == "MockCrewLLM.call"])
    assert r.status == "answered"  # CrewAI retried the Composer task after the transient error
    assert retried == normal + 1  # the failed attempt's prompt was charged too


def test_a_budget_stop_is_sticky():
    b = RequestBudget(cap=10)
    with pytest.raises(BudgetExceeded):
        b.charge(11)
    assert b.exhausted and b.used == 0
    for tokens in (1, 0):  # once stopped, no later call gets through, however small
        with pytest.raises(BudgetExceeded):
            b.charge(tokens)
    assert b.used == 0


def test_a_budget_stop_inside_autogen_surfaces_as_budget_exceeded(collections, monkeypatch):
    tool = RagLookupTool(collection=collections[RECOMMENDED_STRATEGY])
    tool._run(SHOES)
    draft = SupportDraft(policy_claims=[PolicyClaim(text=FOOTWEAR, source_doc_ids=["KB-01"])])
    monkeypatch.setattr(autogen_review, "review_token_reservation", lambda task: 10)  # smaller than a prompt
    budget = RequestBudget()
    with pytest.raises(BudgetExceeded):  # not AutoGen's RuntimeError wrapper
        review_policy_draft(draft, tool.outputs[-1], budget)
    assert budget.used == 0  # the reservation was released; the reviewer's call was refused before it ran
