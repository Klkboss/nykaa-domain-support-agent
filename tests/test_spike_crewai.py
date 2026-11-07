from __future__ import annotations

import json

from crewai.events.listeners.tracing.utils import should_enable_tracing
from crewai.telemetry.telemetry import Telemetry

from app.core.mock_llm import OBSERVATION_MARKER, MockCrewLLM
from tests.spike_support import build_lookup_crew, tool_log


def test_telemetry_and_tracing_disabled():
    assert Telemetry._is_telemetry_disabled() is True
    assert should_enable_tracing() is False


def test_crew_kickoff_invokes_real_tool_chosen_by_schema():
    tool_log.clear()
    llm = MockCrewLLM()
    out = build_lookup_crew(llm).kickoff(inputs={"history": "(none)", "question": "Where is my order ORD-1001?"})
    print("\nCREW RAW OUTPUT:", out.raw)

    # The real tool ran once, with the id from the prompt; the decoy named "order_*" did not.
    assert tool_log == [("check_order_status", {"record_id": "ORD-1001"})]
    assert json.loads(out.raw) == {"status": "Shipped", "order_value_inr": 1499.0}
    assert len(llm.calls) == 2  # Action turn, then Final Answer turn

    # Pitfall A evidence: the system prompt already contains "Observation:" on the FIRST call,
    # yet the mock still chose to call the tool, because it reads assistant turns only.
    first = llm.calls[0]["messages"]
    assert any("Observation:" in m["content"] for m in first if m["role"] == "system")
    assert "Action: check_order_status" in llm.calls[0]["response"]
    second = llm.calls[1]["messages"]
    assert any(OBSERVATION_MARKER in m["content"] for m in second if m["role"] == "assistant")


def test_no_record_id_means_no_tool_call():
    tool_log.clear()
    out = build_lookup_crew(MockCrewLLM(), with_decoy=False).kickoff(
        inputs={"history": "(none)", "question": "Can you check my order?"}
    )
    assert tool_log == []
    assert "order ID" in out.raw
