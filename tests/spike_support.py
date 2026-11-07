"""Compatibility-spike fixtures: a real CrewAI tool, a decoy tool and a one-agent crew.

Replaced by app/tools + app/agents in Phases 4-5.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

from app.core.mock_llm import MockCrewLLM

SPIKE_ORDERS = {
    "ORD-1001": {"status": "Shipped", "order_value_inr": 1499.0},
    "ORD-1002": {"status": "Delivered", "order_value_inr": 2399.0},
}
tool_log: list[tuple[str, dict[str, Any]]] = []


class RecordIdInput(BaseModel):
    record_id: str = Field(..., description="Order record id")


class QueryInput(BaseModel):
    query: str = Field(..., description="Policy search text")


class CheckOrderStatusTool(BaseTool):
    name: str = "check_order_status"
    description: str = "Look up the status and value of one order by its record id."
    args_schema: type[BaseModel] = RecordIdInput

    def _run(self, record_id: str) -> str:
        tool_log.append((self.name, {"record_id": record_id}))
        return json.dumps(SPIKE_ORDERS.get(record_id, {"error": f"unknown record {record_id}"}))


class OrderPolicySearchTool(BaseTool):
    # Decoy: its name also contains "order", so name-substring dispatch would be ambiguous.
    name: str = "order_policy_search"
    description: str = "Search order-related policy text."
    args_schema: type[BaseModel] = QueryInput

    def _run(self, query: str) -> str:
        tool_log.append((self.name, {"query": query}))
        return "Simulated policy text."


def build_lookup_crew(llm: MockCrewLLM, with_decoy: bool = True) -> Crew:
    tools: list[BaseTool] = [OrderPolicySearchTool(), CheckOrderStatusTool()] if with_decoy else [CheckOrderStatusTool()]
    agent = Agent(
        role="Order Lookup Agent",
        goal="Report the status of the customer's order using the order lookup tool.",
        backstory="You only answer from tool results.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=4,
    )
    task = Task(
        description="Conversation so far:\n{history}\n\nCustomer question: {question}",
        expected_output="The order status from the lookup tool, or a request for the order ID.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
