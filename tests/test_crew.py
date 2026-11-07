"""Spec Task 7: three-agent CrewAI crew, real .kickoff() under MOCK_LLM, real tools, network blocked."""

from __future__ import annotations

import json

import pytest

from app.agents import crew as crew_module
from app.agents.crew import NO_HISTORY, build_support_crew, run_support_crew
from app.core.mock_llm import ASK_FOR_ORDER_ID, CONTEXT_MARKER, NO_LOOKUP_NEEDED, MockCrewLLM
from app.rag.generation import FALLBACK_ANSWER, RECOMMENDED_STRATEGY, SIMILARITY_THRESHOLDS, answer_query
from app.schemas import SupportDraft, render_reply
from app.tools import crew_tools
from app.tools.crew_tools import CheckOrderStatusTool, RagLookupTool
from app.tools.order_lookup import ESCALATION_THRESHOLD, check_order_status

STATUS_WORDS = ("Placed", "Shipped", "Delivered", "Returned", "Refunded")
DEMO_A = "How long do I have to return shoes?"
DEMO_B = "What is the status of my order ORD-1016?"
DEMO_C = "My order ORD-1029 arrived late. Can I still return the shoes, and is my case being escalated?"


@pytest.fixture
def rag_tool(collections):
    return RagLookupTool(collection=collections[RECOMMENDED_STRATEGY])


def run(question, rag_tool):
    return run_support_crew(question, rag_tool=rag_tool)


def reply(r) -> str:
    """Phase 6: the crew returns a validated SupportDraft; the customer text is rendered from it in code."""
    assert isinstance(r.structured, SupportDraft)
    return render_reply(r.structured)


def user_text(call) -> str:
    return "\n".join(m["content"] for m in call["messages"] if m["role"] == "user")


def order_sentence(record_id: str) -> str:
    r = check_order_status(record_id)
    decision = ("meets the escalation threshold, so this case is flagged for escalation" if r["escalation_recommended"]
                else "is below the escalation threshold, so no escalation is needed")
    return (f"Order information ({record_id}): status {r['status']}, order value ₹{r['order_value_inr']:,}. "
            f"Escalation score {r['escalation_score']} (threshold {ESCALATION_THRESHOLD}) {decision}.")


# --- Configuration and access boundaries ---------------------------------------------------------

def test_three_agents_with_exact_tool_assignment(rag_tool):
    support = build_support_crew(rag_tool)
    agents = {a.role: a for a in support.crew.agents}
    assert list(agents) == ["Retrieval Agent", "Lookup Agent", "Response Composer"]
    assert [t.name for t in agents["Retrieval Agent"].tools] == ["rag_lookup"]
    assert [t.name for t in agents["Lookup Agent"].tools] == ["check_order_status"]
    assert agents["Response Composer"].tools == []
    holders = [a.role for a in support.crew.agents if any(t.name == "check_order_status" for t in a.tools)]
    assert holders == ["Lookup Agent"]  # least autonomy
    assert all(isinstance(a.llm, MockCrewLLM) for a in support.crew.agents)
    assert all(a.allow_delegation is False for a in support.crew.agents)


def test_task_dependencies(rag_tool):
    retrieval, lookup, compose = build_support_crew(rag_tool).crew.tasks
    assert retrieval.agent.role == "Retrieval Agent" and lookup.agent.role == "Lookup Agent"
    assert lookup.context == []  # sees only the question
    assert compose.agent.role == "Response Composer" and compose.context == [retrieval, lookup]


def test_tool_schemas_drive_dispatch_not_names():
    assert set(RagLookupTool().args_schema.model_fields) == {"query"}
    assert set(CheckOrderStatusTool().args_schema.model_fields) == {"record_id"}
    assert "lookup" in RagLookupTool().name  # spec pitfall: a name-substring router would misroute this


def test_real_llm_mode_is_refused(monkeypatch):
    monkeypatch.setattr(crew_module.config, "MOCK_LLM", False)
    with pytest.raises(NotImplementedError):
        build_support_crew()


# --- Tool wrappers ------------------------------------------------------------------------------------

def test_rag_tool_output_is_the_phase3_answer(collections, rag_tool):
    out = json.loads(rag_tool._run(DEMO_A))
    ref = answer_query(DEMO_A, collections[RECOMMENDED_STRATEGY], SIMILARITY_THRESHOLDS[RECOMMENDED_STRATEGY])
    assert out["answer"] == ref.answer and out["sources"] == ref.sources and out["grounded"] is True
    assert out["threshold"] == 0.33 and all(c["similarity"] >= 0.33 for c in out["context"])
    assert rag_tool.invocations == [{"query": DEMO_A}]


def test_order_tool_outputs():
    tool = CheckOrderStatusTool()
    ok = json.loads(tool._run("ORD-1016"))
    assert ok == {"tool": "check_order_status", "record_id": "ORD-1016", **check_order_status("ORD-1016"),
                  "escalation_threshold": 0.78}
    assert json.loads(tool._run("ORD-9999")) == {"tool": "check_order_status",
                                                 "error": "No order found with record_id ORD-9999"}
    bad = json.loads(tool._run("ORD-1016; DROP TABLE"))
    assert set(bad) == {"tool", "error"} and "DROP" not in bad["error"]


def test_tool_failures_return_generic_errors(monkeypatch):
    class Broken:
        def query(self, **kwargs):
            raise RuntimeError("internal chroma path C:/secret")

    rag = json.loads(RagLookupTool(collection=Broken())._run("refund?"))
    assert rag == {"tool": "rag_lookup", "error": "Policy search is temporarily unavailable."}
    monkeypatch.setattr(crew_tools, "check_order_status", lambda _: (_ for _ in ()).throw(RuntimeError("db secret")))
    order = json.loads(CheckOrderStatusTool()._run("ORD-1016"))
    assert order == {"tool": "check_order_status", "error": "Order lookup is temporarily unavailable."}


# --- Real kickoff: the three demonstrations ----------------------------------------------------------

def test_demo_a_policy_query_invokes_only_rag(rag_tool):
    r = run(DEMO_A, rag_tool)
    assert r.rag_invocations == [{"query": DEMO_A}] and r.order_invocations == []
    assert json.loads(r.task_outputs["retrieval"])["sources"] == ["KB-01"]
    assert r.task_outputs["lookup"] == NO_LOOKUP_NEEDED
    assert reply(r) == ("Policy information (source: KB-01): Footwear can be returned within 30 days of delivery "
                       "if it is unworn and sent back in its original shoebox.")


def test_demo_b_order_query_uses_real_lookup_result(rag_tool):
    r = run(DEMO_B, rag_tool)
    assert r.order_invocations == [{"record_id": "ORD-1016"}]
    assert json.loads(r.task_outputs["lookup"]) == json.loads(CheckOrderStatusTool()._run("ORD-1016"))
    assert reply(r) == order_sentence("ORD-1016")
    assert "Returned" in reply(r) and "₹8,504" in reply(r) and "0.9467" in reply(r) and "flagged for escalation" in reply(r)


def test_demo_c_combined_query_invokes_both_tools(rag_tool):
    r = run(DEMO_C, rag_tool)
    assert r.rag_invocations == [{"query": DEMO_C}]  # the RAG tool got the question, never a record_id
    assert r.order_invocations == [{"record_id": "ORD-1029"}]  # the order tool got only the id
    policy, order = reply(r).split("\n")
    assert policy.startswith("Policy information (source: KB-01") and "30 days" in policy
    assert order == order_sentence("ORD-1029")


def test_composer_receives_both_upstream_outputs(rag_tool):
    support = build_support_crew(rag_tool)
    out = support.crew.kickoff(inputs={"question": DEMO_C, "history": NO_HISTORY})
    composer_prompt = user_text(support.llms["composer"].calls[0])
    context = composer_prompt.split(CONTEXT_MARKER, 1)[1]
    assert out.tasks_output[0].raw in context and out.tasks_output[1].raw in context
    assert len(support.llms["composer"].calls) == 1


def test_lookup_agent_never_sees_policy_output(rag_tool):
    support = build_support_crew(rag_tool)
    support.crew.kickoff(inputs={"question": DEMO_C, "history": NO_HISTORY})
    for call in support.llms["lookup"].calls:
        assert '"tool": "rag_lookup"' not in user_text(call) and CONTEXT_MARKER not in user_text(call)


def test_observation_in_system_prompt_is_not_mistaken_for_a_tool_result(rag_tool):
    support = build_support_crew(rag_tool)
    support.crew.kickoff(inputs={"question": DEMO_A, "history": NO_HISTORY})
    first = support.llms["retrieval"].calls[0]
    assert any("Observation:" in m["content"] for m in first["messages"] if m["role"] == "system")
    assert "Action: rag_lookup" in first["response"]  # it still acted, instead of "answering" with template text


# --- Edge cases ------------------------------------------------------------------------------------------

def test_unknown_order_never_invents_details(rag_tool):
    r = run("Where is my order ORD-9999?", rag_tool)
    assert r.order_invocations == [{"record_id": "ORD-9999"}]
    assert reply(r) == "Order information: No order found with record_id ORD-9999. No order details have been assumed."
    assert not any(w in reply(r) for w in STATUS_WORDS) and "₹" not in reply(r) and "score" not in reply(r)


@pytest.mark.parametrize("question", ["Can you check my order status?", "Check order ORD-12 please"])
def test_missing_or_malformed_id_asks_for_it_without_calling_the_tool(rag_tool, question):
    r = run(question, rag_tool)
    assert r.order_invocations == [] and r.task_outputs["lookup"] == ASK_FOR_ORDER_ID
    assert reply(r) == ASK_FOR_ORDER_ID


def test_lower_case_id_is_still_looked_up(rag_tool):
    r = run("status of ord-1016?", rag_tool)
    assert r.order_invocations == [{"record_id": "ord-1016"}] and reply(r) == order_sentence("ORD-1016")


def test_out_of_scope_query_returns_fallback(rag_tool):
    r = run("What is the best way to train for a marathon?", rag_tool)
    assert json.loads(r.task_outputs["retrieval"])["grounded"] is False
    assert r.order_invocations == [] and reply(r) == FALLBACK_ANSWER


def test_policy_search_failure_reaches_the_customer_safely(rag_tool, collections):
    class Broken:
        def query(self, **kwargs):
            raise RuntimeError("secret internals")

    r = run(DEMO_A, RagLookupTool(collection=Broken()))
    assert reply(r) == "Policy information: Policy search is temporarily unavailable." and "secret" not in reply(r)


def test_order_lookup_failure_reaches_the_customer_safely(rag_tool, monkeypatch):
    monkeypatch.setattr(crew_tools, "check_order_status", lambda _: (_ for _ in ()).throw(RuntimeError("db secret")))
    r = run(DEMO_B, rag_tool)
    assert reply(r) == ("Order information: Order lookup is temporarily unavailable. "
                       "No order details have been assumed.")


@pytest.mark.parametrize("record_id, flagged", [("ORD-1016", True), ("ORD-1045", True), ("ORD-1031", False),
                                                 ("ORD-1038", False), ("ORD-1001", False)])
def test_escalation_decision_propagates_unchanged(rag_tool, record_id, flagged):
    r = run(f"What is the status of {record_id}?", rag_tool)
    assert reply(r) == order_sentence(record_id)
    assert ("flagged for escalation" in reply(r)) is flagged is check_order_status(record_id)["escalation_recommended"]


def test_kickoff_is_deterministic(rag_tool, collections):
    first = run(DEMO_C, rag_tool)
    second = run(DEMO_C, RagLookupTool(collection=collections[RECOMMENDED_STRATEGY]))
    assert (first.raw, first.task_outputs, first.llm_calls) == (second.raw, second.task_outputs, second.llm_calls)


def test_llm_call_counts_show_real_tool_loops(rag_tool):
    r = run(DEMO_C, rag_tool)
    assert r.llm_calls == {"retrieval": 2, "lookup": 2, "composer": 1}  # action + final for each tool agent
