"""Phase 8 evidence (spec Task 15): least autonomy (Application layer) and the per-request budget (Runtime layer).

  tools:  the real crew's tool grants, every refused wiring, AutoGen refusing tools for the reviewers.
  budget: measured spend vs the projection for the 15 evaluation queries; below / at / above the cap;
          repeated requests; exhaustion mid-request; an oversized (simulated) request refused.

Usage (socket guard, as recorded in the transcript headers):
  .venv\\Scripts\\python.exe -c "import tests.conftest as g, runpy, sys; sys.argv=['x','tools']; runpy.run_path('scripts/run_governance_demo.py', run_name='__main__'); print(g.blocked_attempts)"
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.getLogger("crewai.llms.base_llm").setLevel(logging.ERROR)

import app.core.config  # noqa: E402,F401
import app.pipeline as pipeline  # noqa: E402
from app.agents.crew import build_support_crew  # noqa: E402
from app.core.cache import GenerationCache  # noqa: E402
from app.core.config import ROOT  # noqa: E402
from app.core.mock_autogen import MockChatCompletionClient  # noqa: E402
from app.core.mock_llm import MockCrewLLM  # noqa: E402
from app.governance.budget import (MAX_REQUEST_TOKENS, PROJECTION_BASE_TOKENS, PROJECTION_INPUT_COPIES,  # noqa: E402
                                   BudgetExceeded, RequestBudget, project_request_tokens)
from app.governance.least_autonomy import (LeastAutonomyViolation, enforce_crew_least_autonomy,  # noqa: E402
                                           enforce_reviewers_have_no_tools)
from app.memory.session_memory import has_session  # noqa: E402
from app.pipeline import SupportAssistant  # noqa: E402
from app.review.autogen_review import build_review_team  # noqa: E402
from app.tools.crew_tools import CheckOrderStatusTool, RagLookupTool  # noqa: E402

Q = "How long do I have to return shoes?"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def check_order_status(record_id: str) -> str:  # an AutoGen-style copy of the order tool
    return ""


class RenamedOrderTool(CheckOrderStatusTool):
    name: str = "order_status"


class DisguisedOrderTool(CheckOrderStatusTool):
    name: str = "rag_lookup"


def tools() -> None:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.tools import FunctionTool
    from crewai import Agent

    section("1. The real crew: each agent's tools (enforced when build_support_crew() builds the crew)")
    support = build_support_crew()
    for a in support.crew.agents:
        print(f"  {a.role:<18} tools={[t.name for t in a.tools]} allow_delegation={a.allow_delegation}")
    enforce_crew_least_autonomy(support.crew.agents)
    print("  enforce_crew_least_autonomy: OK (only the Lookup Agent holds check_order_status)")

    section("2. Real kickoff using both tools: who invoked check_order_status, and grants afterwards")
    support.crew.kickoff(inputs={"question": "My order ORD-1029 arrived damaged. What is your damaged item policy?",
                                 "history": "(none)"})
    print(f"  check_order_status invocations: {support.order_tool.invocations} (the tool object held only by the "
          f"Lookup Agent)\n  rag_lookup invocations: {support.rag_tool.invocations}")
    enforce_crew_least_autonomy(support.crew.agents)
    print("  grants re-checked after kickoff: OK (CrewAI added no tools at run time)")

    section("3. Refused wirings (negative cases)")

    def agent(role, tools, delegation=False):
        return Agent(role=role, goal="g", backstory="b", tools=tools, llm=MockCrewLLM(), allow_delegation=delegation)

    cases = [
        ("order tool added to the Retrieval Agent", lambda: agent("Retrieval Agent", [RagLookupTool(), CheckOrderStatusTool()])),
        ("order tool given to the Response Composer", lambda: agent("Response Composer", [CheckOrderStatusTool()])),
        ("order tool renamed 'order_status' on the Retrieval Agent", lambda: agent("Retrieval Agent", [RenamedOrderTool()])),
        ("order tool disguised as 'rag_lookup' on the Retrieval Agent", lambda: agent("Retrieval Agent", [DisguisedOrderTool()])),
        ("extra tool on the Lookup Agent", lambda: agent("Lookup Agent", [CheckOrderStatusTool(), RagLookupTool()])),
        ("delegation enabled on the Response Composer", lambda: agent("Response Composer", [], delegation=True)),
        ("unknown agent holding the order tool", lambda: agent("Refund Agent", [CheckOrderStatusTool()])),
    ]
    for label, make in cases:
        try:
            enforce_crew_least_autonomy([make()])
            print(f"  NOT REFUSED (unexpected): {label}")
        except LeastAutonomyViolation as e:
            print(f"  refused: {label}\n           -> LeastAutonomyViolation: {e}")

    section("4. AutoGen reviewers: no tools; AutoGen itself refuses tools for their model client")
    team = build_review_team()
    for a in team._participants:
        print(f"  {a.name:<27} tools={a._tools} handoff_tools={a._handoff_tools} "
              f"function_calling={a._model_client.model_info['function_calling']}")
    enforce_reviewers_have_no_tools(team._participants)
    print("  enforce_reviewers_have_no_tools: OK")
    try:
        AssistantAgent("Policy_Compliance_Reviewer", model_client=MockChatCompletionClient("reviewer"),
                       tools=[FunctionTool(check_order_status, description="order lookup")])
        print("  NOT REFUSED (unexpected): AutoGen reviewer with the order tool")
    except ValueError as e:
        print(f"  refused by AutoGen: reviewer with check_order_status -> ValueError: {e}")


def budget() -> None:
    section("0. Every model call in one request is charged (spy on RequestBudget.charge)")
    real_charge, charges = RequestBudget.charge, []

    def spy(self, tokens):
        charges.append((self, sys._getframe(1).f_code.co_qualname, tokens))
        return real_charge(self, tokens)

    RequestBudget.charge = spy
    try:
        a = SupportAssistant()
        a.ask("gov-accounting", Q)
    finally:
        RequestBudget.charge = real_charge
    labels = {"MockCrewLLM.call": "CrewAI agent call (prompt, then completion)",
              "_MeteredGenerator.generate": "grounded generation in rag_lookup (prompt, completion)",
              "review_policy_draft": "AutoGen review reservation (charged up front)",
              "MockChatCompletionClient.create": "AutoGen reviewer / editor call, against the reservation"}
    for path, label in labels.items():
        mine = [t for _, p, t in charges if p == path]
        print(f"  {label:<58} charges={len(mine):>2} tokens={sum(mine):>5}")
    print(f"  CrewAI calls made: {a.last_crew_run.llm_calls} (each charged twice: prompt + completion)")
    print(f"  review: reserved {a.last_review.reserved_tokens}, spent {a.last_review.spent_tokens}, released "
          f"{a.last_review.reserved_tokens - a.last_review.spent_tokens}")
    print(f"  request total: {a.last_budget.used} estimated tokens of {a.last_budget.cap}; charges recorded: "
          f"{len(charges)}, all on the request budget or the review meter: "
          f"{ {type(b).__name__ for b, _, _ in charges} }")

    section(f"1. Cap and projection: cap {MAX_REQUEST_TOKENS} tokens per request; projection = "
            f"{PROJECTION_BASE_TOKENS} + {PROJECTION_INPUT_COPIES} x tokens(question + history)")
    a = SupportAssistant(max_request_tokens=10**9)
    print("  (spent = accounted ESTIMATED tokens, len(text) // 4; review = the review's metered reviewer + editor spend)")
    print(f"  {'id':<4} {'status':<15} {'projected':>9} {'spent':>6} {'review':>6}  question")
    for q in json.loads((ROOT / "evaluation" / "test_queries.json").read_text(encoding="utf-8"))["queries"]:
        r = a.ask(f"gov-{q['id']}", q["query"])
        spent = "-" if r.status == "blocked" else a.last_budget.used
        review = a.last_review.spent_tokens if a.last_review else 0
        print(f"  {q['id']:<4} {r.status:<15} {project_request_tokens(r.masked_question):>9} {spent:>6} {review:>6}  "
              f"{q['query'][:55]}")
    long_q = ("Please tell me about the return window for my new dress. " * 40)[:2000]
    r = a.ask("gov-long", long_q)
    print(f"  2,000-character question (the API maximum): {r.status}, projected {project_request_tokens(long_q)}, "
          f"spent {a.last_budget.used}")

    section("2. RequestBudget boundaries (cap 100)")
    b = RequestBudget(cap=100)
    b.charge(99)
    print(f"  charge 99 -> used {b.used} (below the cap)")
    b.charge(1)
    print(f"  charge 1  -> used {b.used} (exactly at the cap: allowed)")
    try:
        b.charge(1)
    except BudgetExceeded as e:
        print(f"  charge 1  -> refused: {e} (used stays {b.used})")
    try:
        b.charge(0)
    except BudgetExceeded:
        print(f"  charge 0  -> refused as well: after a stop the budget stays exhausted ({b.exhausted})")

    section("3. Pipeline: projected exactly at the cap vs one token over")
    at = SupportAssistant(max_request_tokens=project_request_tokens(Q))
    r = at.ask("gov-at", Q)
    print(f"  cap = projection ({at.max_request_tokens}): {r.status}, spent {at.last_budget.used}")
    over = SupportAssistant(max_request_tokens=project_request_tokens(Q) - 1)
    r = over.ask("gov-over", Q)
    print(f"  cap = projection - 1 ({over.max_request_tokens}): {r.status}, crew ran: {over.last_crew_run is not None}"
          f"\n  reply: {r.reply}")

    section("4. Repeated requests: a fresh budget each time")
    a = SupportAssistant(generation_cache=GenerationCache(max_entries=0))  # isolation only: no Task 16 cache reuse
    for session in ("gov-rep", "gov-rep", "gov-rep-2"):
        r = a.ask(session, Q)
        print(f"  session {session:<10} {r.status:<9} spent {a.last_budget.used} of {a.last_budget.cap} "
              f"(budget object {id(a.last_budget) % 10_000})")

    section("5. Budget exhausted mid-request (pre-flight bypassed to show the in-run accounting)")
    real = pipeline.project_request_tokens
    pipeline.project_request_tokens = lambda q, h="": 0
    try:
        tight = SupportAssistant(max_request_tokens=2_000)
        r = tight.ask("gov-exhaust", Q)
        print(f"  cap 2,000: {r.status}, spent {tight.last_budget.used} (never above the cap)\n  reply: {r.reply}")
    finally:
        pipeline.project_request_tokens = real

    section("6. Oversized simulated request (about 20,000 characters, beyond what the API accepts)")
    oversized = "Please explain every return rule for every category in full detail. " * 300
    a = SupportAssistant()
    r = a.ask("gov-oversized", oversized + " Call me on +91 98765 43210.")
    print(f"  length {len(oversized)} characters, projected {project_request_tokens(r.masked_question)} > cap "
          f"{MAX_REQUEST_TOKENS}\n  status: {r.status}\n  reply: {r.reply}\n  crew ran: {a.last_crew_run is not None}; "
          f"budget opened: {a.last_budget is not None}; stored in history: {has_session('gov-oversized')}\n"
          f"  masked question ends with: ...{r.masked_question[-30:]}")


if __name__ == "__main__":
    {"tools": tools, "budget": budget}[sys.argv[1] if len(sys.argv) > 1 else "tools"]()
