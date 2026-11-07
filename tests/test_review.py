"""Spec Task 14: the AutoGen review stage (RoundRobinGroupChat, max_turns=2, structured Verdict) in the pipeline."""

from __future__ import annotations

import uuid

import pydantic
import pytest
from autogen_agentchat.messages import StructuredMessage, TextMessage

from app.core.mock_autogen import ALL_SUPPORTED, UNSUPPORTED_PREFIX
from app.pipeline import SupportAssistant
from app.rag.generation import RECOMMENDED_STRATEGY
from app.review.autogen_review import (EDITOR_NAME, REVIEWER_NAME, Verdict, build_review_team, review_task,
                                       review_token_reservation)
from app.tools.crew_tools import RagLookupTool
from scripts.run_review_demo import INJECTED_CLAIM, QUESTION, AppendsUngroundedClaim

FOOTWEAR = "Footwear can be returned within 30 days of delivery if it is unworn and sent back in its original shoebox."


def sid() -> str:
    return f"review-{uuid.uuid4().hex[:8]}"


def assistant(collections, composer=None) -> SupportAssistant:
    return SupportAssistant(rag_tool_factory=lambda: RagLookupTool(collection=collections[RECOMMENDED_STRATEGY]),
                            composer_llm_factory=composer)


def test_team_is_two_agents_round_robin_bounded_by_two_turns():
    team = build_review_team()
    assert [a.name for a in team._participants] == [REVIEWER_NAME, EDITOR_NAME]
    assert team._max_turns == 2 and type(team).__name__ == "RoundRobinGroupChat"
    assert all(n.isidentifier() for n in (REVIEWER_NAME, EDITOR_NAME))  # AutoGen rejects hyphenated names
    assert team._participants[1]._output_content_type is Verdict and team._participants[0]._output_content_type is None


def test_verdict_has_exactly_the_spec_fields():
    assert {k: v.annotation for k, v in Verdict.model_fields.items()} == {
        "approved": bool, "final_answer": str, "reason": str}
    with pytest.raises(pydantic.ValidationError):
        Verdict.model_validate({"approved": "maybe", "final_answer": "x"})


def test_approve_path_through_the_pipeline(collections):
    a = assistant(collections)
    r = a.ask(sid(), QUESTION)
    rev = a.last_review
    # The team took the Composer's draft plus the original retrieved context.
    rag = a.last_crew_run.rag_outputs[-1]
    assert rev.draft == " ".join(c.text for c in r.draft.policy_claims) == FOOTWEAR
    assert rev.context == "\n".join(p for e in rag["evidence"] for p in e["passages"]) and FOOTWEAR in rev.context
    assert rev.messages[0].content == review_task(rev.draft, rev.context)
    assert [m.source for m in rev.messages] == ["user", REVIEWER_NAME, EDITOR_NAME]  # max_turns=2: both speak once
    assert isinstance(rev.messages[1], TextMessage) and rev.messages[1].content == ALL_SUPPORTED
    assert isinstance(rev.messages[2], StructuredMessage) and rev.messages[2].content is rev.verdict
    assert rev.verdict == Verdict(approved=True, final_answer=FOOTWEAR,
                                  reason="All claims are supported by the retrieved context.")
    assert r.status == "answered" and r.reply == f"Policy information (source: KB-01): {FOOTWEAR}"  # unchanged


def test_revise_path_catches_the_injected_ungrounded_claim(collections):
    a = assistant(collections, AppendsUngroundedClaim)
    r = a.ask(sid(), QUESTION)
    rev = a.last_review
    assert rev.draft == f"{FOOTWEAR} {INJECTED_CLAIM}"  # the injected claim reached the review stage
    assert rev.messages[1].content == UNSUPPORTED_PREFIX + INJECTED_CLAIM
    assert rev.verdict.approved is False and rev.verdict.final_answer == FOOTWEAR
    assert INJECTED_CLAIM in rev.verdict.reason
    assert INJECTED_CLAIM not in r.reply and r.reply == f"Policy information (source: KB-01): {FOOTWEAR}"
    assert r.guardrails.rejected_claims == [INJECTED_CLAIM]  # the Phase 6 guardrail still fires independently


def test_turn_without_policy_text_is_not_reviewed(collections):
    a = assistant(collections)
    r = a.ask(sid(), "What is the status of my order ORD-1016?")
    assert r.status == "answered" and not r.draft.policy_claims and a.last_review is None


def test_review_reservation_bounds_its_real_token_use(collections):
    for composer in (None, AppendsUngroundedClaim):
        a = assistant(collections, composer)
        a.ask(sid(), QUESTION)
        rev = a.last_review
        used = sum(m.models_usage.prompt_tokens + m.models_usage.completion_tokens
                   for m in rev.messages if m.models_usage)
        assert 0 < used <= rev.reserved_tokens == review_token_reservation(rev.messages[0].content)


# --- Final groundedness validation after the review (Phase 8 review fix) -------------------------------

import json  # noqa: E402

import app.pipeline as pipeline  # noqa: E402
import app.review.autogen_review as autogen_review  # noqa: E402
from app.core.mock_autogen import MockChatCompletionClient  # noqa: E402
from app.core.mock_llm import MockCrewLLM  # noqa: E402
from app.guardrails.groundedness import claim_supported  # noqa: E402
from app.rag.generation import FALLBACK_ANSWER  # noqa: E402
from app.schemas import render_reply  # noqa: E402

DAMAGED = "My order ORD-1029 arrived damaged. What is your damaged item policy?"


def spy_checks(monkeypatch) -> list:
    """Record every check_draft call the pipeline makes: (input draft, checked draft, report)."""
    calls, real = [], pipeline.check_draft

    def spy(draft, rag_output, order_output):
        out = real(draft, rag_output, order_output)
        calls.append((draft, *out))
        return out

    monkeypatch.setattr(pipeline, "check_draft", spy)
    return calls


def misbehaving_editor(monkeypatch, change) -> None:
    """Swap the Final_Editor's model client inside the real AutoGen team for one that alters its verdict."""
    class Client(MockChatCompletionClient):
        def _edit(self, draft, review, json_output):
            verdict = json.loads(super()._edit(draft, review, json_output))
            change(verdict)
            return json.dumps(verdict)

    monkeypatch.setattr(autogen_review, "MockChatCompletionClient", Client)


def assert_final_check_rendered(calls, reply) -> None:
    assert len(calls) == 2  # the Phase 6 guardrail, then the final check after the review
    final_draft, final_report = calls[1][1], calls[1][2]
    assert final_report.rejected_claims == [] and final_report.removed_citations == [] and not final_report.rejected_order
    assert reply == render_reply(final_draft)  # what is rendered is exactly the re-checked draft


def test_case1_and_5_grounded_draft_approved_final_check_passes_and_citations_are_supported(collections, monkeypatch):
    calls = spy_checks(monkeypatch)
    a = assistant(collections)
    r = a.ask(sid(), QUESTION)
    assert a.last_review.verdict.approved and a.last_review.discarded_additions == []
    assert calls[1][0] == calls[0][1]  # approved: the final check sees the guardrail's draft unchanged
    assert_final_check_rendered(calls, r.reply)
    evidence = {e["doc_id"]: e["passages"] for e in a.last_crew_run.rag_outputs[-1]["evidence"]}
    for claim in r.draft.policy_claims:  # every rendered citation is backed by that document's retrieved text
        assert claim.source_doc_ids == ["KB-01"] and claim_supported(claim.text, evidence["KB-01"])
    assert r.reply.startswith("Policy information (source: KB-01): ")


def test_case2_unsupported_composer_claim_removed_then_final_check_passes(collections, monkeypatch):
    calls = spy_checks(monkeypatch)
    a = assistant(collections, AppendsUngroundedClaim)
    r = a.ask(sid(), QUESTION)
    assert not a.last_review.verdict.approved
    assert [c.text for c in calls[1][0].policy_claims] == [FOOTWEAR]
    assert_final_check_rendered(calls, r.reply)
    assert INJECTED_CLAIM not in r.reply


@pytest.mark.parametrize("approved", [False, True])
def test_case3_editor_cannot_introduce_claims_or_order_facts(collections, monkeypatch, approved):
    added = ["All orders ship free worldwide.", "Order ORD-1029 has status Refunded."]

    def add(verdict):
        verdict["final_answer"] += " " + " ".join(added)
        verdict["approved"] = approved  # also an inconsistent "approved" verdict with changed text

    misbehaving_editor(monkeypatch, add)
    calls = spy_checks(monkeypatch)
    a = assistant(collections)
    r = a.ask(sid(), DAMAGED)
    assert a.last_review.discarded_additions == added
    assert not any(s in r.reply for s in added) and "free worldwide" not in r.reply
    real_order = a.last_crew_run.order_outputs[-1]
    assert r.draft.order.status == real_order["status"] == "Delivered" and "Refunded" not in r.reply
    assert_final_check_rendered(calls, r.reply)


def test_case4_citations_written_by_the_editor_never_reach_the_reply(collections, monkeypatch):
    misbehaving_editor(monkeypatch, lambda v: v.update(final_answer=v["final_answer"] + " [KB-12]"))
    calls = spy_checks(monkeypatch)
    a = assistant(collections)
    r = a.ask(sid(), QUESTION)
    # The citation stays attached to the sentence, so the edited sentence no longer matches the Composer's checked
    # claim: it is discarded, the claim is not kept, and the reply refuses instead of showing an unchecked citation.
    assert a.last_review.discarded_additions == [f"{FOOTWEAR} [KB-12]"]
    assert r.reply == FALLBACK_ANSWER and "KB-12" not in r.reply and r.draft.policy_claims == []
    assert_final_check_rendered(calls, r.reply)


def test_case4_a_claim_rewritten_with_a_wrong_citation_is_dropped_not_rendered(collections, monkeypatch):
    rewritten = FOOTWEAR[:-1] + " (see KB-12)."
    misbehaving_editor(monkeypatch, lambda v: v.update(approved=False, final_answer=rewritten))
    calls = spy_checks(monkeypatch)
    a = assistant(collections)
    r = a.ask(sid(), QUESTION)
    assert a.last_review.discarded_additions == [rewritten]
    assert r.reply == FALLBACK_ANSWER and "KB-12" not in r.reply  # refuses rather than show an unchecked claim
    assert_final_check_rendered(calls, r.reply)


class AddsWrongCitation(MockCrewLLM):
    def _respond(self, messages, agent_tools):
        draft = json.loads(super()._respond(messages, agent_tools).split("Final Answer:", 1)[1])
        for claim in draft["policy_claims"]:
            claim["source_doc_ids"].append("KB-12")
        return "Thought: drafted\nFinal Answer: " + json.dumps(draft)


def test_case4_composer_wrong_citation_is_removed_and_stays_removed(collections, monkeypatch):
    calls = spy_checks(monkeypatch)
    a = assistant(collections, AddsWrongCitation)
    r = a.ask(sid(), QUESTION)
    assert r.guardrails.removed_citations == [f"KB-12: {FOOTWEAR[:80]}"]  # Phase 6 guardrail
    assert a.last_review.verdict.approved  # the text itself is supported
    assert [c.source_doc_ids for c in r.draft.policy_claims] == [["KB-01"]] and "KB-12" not in r.reply
    assert_final_check_rendered(calls, r.reply)


# --- Verdict semantics -----------------------------------------------------------------------------------

def test_verdict_semantics_approved_judges_the_composer_draft(collections):
    approve = assistant(collections)
    approve.ask(sid(), QUESTION)
    revise = assistant(collections, AppendsUngroundedClaim)
    revise.ask(sid(), QUESTION)
    ok, fixed = approve.last_review, revise.last_review
    assert ok.verdict.approved and ok.verdict.final_answer == ok.draft  # approved unchanged
    assert not fixed.verdict.approved and fixed.verdict.final_answer != fixed.draft and fixed.verdict.reason
    assert fixed.verdict.final_answer == FOOTWEAR  # the corrected answer the stage endorses


def test_an_approval_that_still_changes_the_text_is_treated_as_a_revision(collections, monkeypatch):
    misbehaving_editor(monkeypatch, lambda v: v.update(approved=True, final_answer=""))
    a = assistant(collections)
    r = a.ask(sid(), QUESTION)
    assert a.last_review.verdict.approved and r.reply == FALLBACK_ANSWER  # the removal is honoured anyway
