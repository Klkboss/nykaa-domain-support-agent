"""RunnableWithMessageHistory wrapping a pipeline that really runs CrewAI and AutoGen."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from app.core.mock_llm import MockCrewLLM
from app.memory.session_memory import clear_sessions, get_session_history, with_session_memory
from app.review.autogen_review import review
from tests.spike_support import build_lookup_crew, tool_log


def _format_history(history: list[BaseMessage]) -> str:
    return "\n".join(f"{m.type}: {m.content}" for m in history) or "(none)"


def spike_pipeline(inputs: dict[str, Any]) -> dict[str, Any]:
    crew_out = build_lookup_crew(MockCrewLLM(), with_decoy=False).kickoff(
        inputs={"history": _format_history(inputs["history"]), "question": inputs["question"]}
    )
    # Review the crew draft against what the tool returned (the only context in this spike).
    verdict, _ = asyncio.run(review(crew_out.raw, crew_out.raw))
    return {"answer": verdict.final_answer, "approved": verdict.approved}


def test_multi_turn_memory_and_session_isolation():
    clear_sessions()
    chat = with_session_memory(RunnableLambda(spike_pipeline))

    def ask(session: str, question: str) -> dict[str, Any]:
        return chat.invoke({"question": question}, config={"configurable": {"session_id": session}})

    ask("session-A", "Hi, my order ID is ORD-1002.")
    tool_log.clear()

    # Turn 2 in the same session: the id is only in memory, not in the question.
    same = ask("session-A", "Can you check its status?")
    assert tool_log == [("check_order_status", {"record_id": "ORD-1002"})]
    assert "Delivered" in same["answer"] and same["approved"] is True

    # Same question in a fresh session: no memory, so no tool call and a request for the id.
    tool_log.clear()
    fresh = ask("session-B", "Can you check its status?")
    assert tool_log == []
    assert "order ID" in fresh["answer"]

    assert len(get_session_history("session-A").messages) == 4  # 2 human + 2 ai
    assert len(get_session_history("session-B").messages) == 2
    print("\nSESSION A:", [m.content for m in get_session_history("session-A").messages])
    print("SESSION B:", [m.content for m in get_session_history("session-B").messages])
