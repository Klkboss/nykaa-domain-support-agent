"""Application governance layer (spec Task 15): least autonomy.

Every agent gets an explicit tool allow-list, checked whenever the crew or the review team is built. Only the
Lookup Agent may hold `check_order_status`; an order-lookup tool is recognised by its name OR its argument schema,
so renaming it does not get it past the guard. Delegation is refused, because CrewAI's delegation tools would let
an agent hand work to another agent's tools. The AutoGen reviewers hold no tools at all; their model client
declares no function calling, so AutoGen itself also refuses tools for them.
"""

from __future__ import annotations

from typing import Any

from app.tools.crew_tools import OrderLookupInput

ORDER_TOOL = "check_order_status"
LOOKUP_ROLE = "Lookup Agent"
ALLOWED_TOOLS: dict[str, set[str]] = {
    "Retrieval Agent": {"rag_lookup"},
    LOOKUP_ROLE: {ORDER_TOOL},
    "Response Composer": set(),
}


class LeastAutonomyViolation(ValueError):
    pass


def is_order_tool(tool: Any) -> bool:
    return getattr(tool, "name", None) == ORDER_TOOL or getattr(tool, "args_schema", None) is OrderLookupInput


def enforce_crew_least_autonomy(agents: list[Any]) -> None:
    for agent in agents:
        allowed = ALLOWED_TOOLS.get(agent.role)
        if allowed is None:
            raise LeastAutonomyViolation(f"{agent.role!r} has no tool grant; every agent needs an explicit allow-list")
        tools = list(agent.tools or [])
        if {t.name for t in tools} != allowed:
            raise LeastAutonomyViolation(f"{agent.role} must hold exactly {sorted(allowed)}, "
                                         f"got {sorted(t.name for t in tools)}")
        if agent.role != LOOKUP_ROLE and any(is_order_tool(t) for t in tools):
            raise LeastAutonomyViolation(f"only the {LOOKUP_ROLE} may hold the order-lookup tool, not {agent.role}")
        if agent.allow_delegation:
            raise LeastAutonomyViolation(f"{agent.role} may not delegate: delegation reaches other agents' tools")


def enforce_reviewers_have_no_tools(agents: list[Any]) -> None:
    for agent in agents:
        # AssistantAgent keeps its tools in _tools / _handoff_tools (autogen-agentchat 0.7.5, pinned).
        if agent._tools or agent._handoff_tools or agent._model_client.model_info["function_calling"]:
            raise LeastAutonomyViolation(f"AutoGen reviewer {agent.name} may not hold tools")
