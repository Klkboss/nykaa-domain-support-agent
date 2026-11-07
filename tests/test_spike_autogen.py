from __future__ import annotations

import asyncio

from autogen_agentchat.messages import StructuredMessage, TextMessage

from app.review.autogen_review import EDITOR_NAME, REVIEWER_NAME, Verdict, review

CONTEXT = (
    "COD refunds are credited to the Nykaa wallet within 5 to 7 business days after the "
    "returned item passes quality check."
)


def _run(draft: str) -> tuple[Verdict, list]:
    verdict, messages = asyncio.run(review(draft, CONTEXT))
    for m in messages:
        print(f"\n[{m.source}] {type(m).__name__}: {m.content}")
    return verdict, messages


def test_grounded_draft_is_approved_unchanged():
    draft = "COD refunds are credited to the Nykaa wallet within 5 to 7 business days after quality check."
    verdict, messages = _run(draft)

    assert [m.source for m in messages] == ["user", REVIEWER_NAME, EDITOR_NAME]  # max_turns=2
    assert isinstance(messages[1], TextMessage)
    assert isinstance(messages[-1], StructuredMessage) and isinstance(messages[-1].content, Verdict)
    assert verdict.approved is True
    assert verdict.final_answer == draft


def test_flawed_draft_is_revised():
    flawed = (
        "COD refunds are credited to the Nykaa wallet within 5 to 7 business days after quality check. "
        "Premium members get instant cash refunds at any store."
    )
    verdict, _ = _run(flawed)

    assert verdict.approved is False
    assert "instant cash" not in verdict.final_answer
    assert verdict.final_answer.startswith("COD refunds are credited")
    assert "Premium members" in verdict.reason


def test_verdict_schema_rejects_bad_payload():
    import pydantic
    import pytest

    with pytest.raises(pydantic.ValidationError):
        Verdict.model_validate({"approved": "maybe", "final_answer": "x"})
