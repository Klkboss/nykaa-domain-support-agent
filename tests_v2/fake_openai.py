"""A fake OpenAI-compatible server on httpx.MockTransport: scripted responses, every request recorded."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import httpx

from app.core.mock_llm import json_objects

BASE_URL = "http://llm.test/v1"  # non-loopback: an unmocked request would hit the network guard


def completion(content: str | None = "ok", usage: tuple[int, int] | None = (11, 3),
               tool_calls: list[dict[str, Any]] | None = None, **reasoning: str) -> dict[str, Any]:
    """`reasoning` adds a reasoning model's separate fields: reasoning=... (Groq) or reasoning_content=..."""
    message: dict[str, Any] = {"role": "assistant", "content": content, **reasoning}
    if tool_calls:
        message["tool_calls"] = tool_calls
    body: dict[str, Any] = {"id": "cmpl-1", "object": "chat.completion", "created": 0, "model": "fake",
                            "choices": [{"index": 0, "message": message,
                                         "finish_reason": "tool_calls" if tool_calls else "stop"}]}
    if usage is not None:
        body["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}
    return body


Step = dict[str, Any] | httpx.Response | Exception


class FakeServer:
    """Answers each request with the next scripted step (a JSON body, a Response, or an exception to raise)."""

    def __init__(self, *steps: Step) -> None:
        self.steps = list(steps)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, Exception):
            raise step
        if isinstance(step, httpx.Response):
            return step
        return httpx.Response(200, json=step | {"model": json.loads(request.content).get("model", "fake")})

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]

    def sync_client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def async_client(self) -> httpx.AsyncClient:
        handler: Callable[[httpx.Request], httpx.Response] = self.handler

        async def ahandler(request: httpx.Request) -> httpx.Response:
            return handler(request)

        return httpx.AsyncClient(transport=httpx.MockTransport(ahandler))


def status(code: int, **headers: str) -> httpx.Response:
    return httpx.Response(code, json={"error": {"message": "fake provider error"}}, headers=headers)


def _contents(body: dict[str, Any]) -> list[str]:
    out = []
    for m in body.get("messages", []):
        c = m.get("content")
        out.extend([p.get("text", "") for p in c] if isinstance(c, list) else [str(c or "")])
    return out


def _schema_name(body: dict[str, Any]) -> str | None:
    fmt = body.get("response_format") or {}
    return (fmt.get("json_schema") or {}).get("name") if isinstance(fmt, dict) else None


def _tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"id": f"call-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class CrewFake:
    """Plays every real-model role of the v2 pipeline, deciding from what each request contains (like a model).

    Roles: Retrieval Agent (calls rag_lookup), Lookup Agent (calls check_order_status only if an order id is in
    the question), grounded generator (quotes sentence 1), Composer (copies the tool JSON into a SupportDraft),
    AutoGen reviewer and editor (approve). `overrides` maps a role to a step (JSON body, Response or exception), or
    to a list of steps used in order (the last one repeats; None = answer normally). Every request and its role are
    recorded; each answer reports usage (11, 3).
    """

    def __init__(self, **overrides: Any) -> None:
        self.overrides = {k: list(v) if isinstance(v, list) else v for k, v in overrides.items()}
        self.roles: list[str] = []
        self.requests: list[dict[str, Any]] = []

    def role(self, body: dict[str, Any]) -> str:
        tools = {t["function"]["name"] for t in body.get("tools", [])}
        schema = _schema_name(body)
        system = next((c for c in _contents(body)[:1]), "")
        if "rag_lookup" in tools:
            return "retrieval"
        if "check_order_status" in tools:
            return "lookup"
        if schema == "grounded_answer":
            return "generator"
        if schema == "SupportDraft":
            return "composer"
        if schema == "Verdict":
            return "editor"
        if "UNSUPPORTED" in system:
            return "reviewer"
        return "unknown"

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        role = self.role(body)
        self.roles.append(role)
        self.requests.append(body)
        step = self.overrides.get(role)
        if isinstance(step, list):
            step = step.pop(0) if len(step) > 1 else step[0]
        if isinstance(step, Exception):
            raise step
        if isinstance(step, httpx.Response):
            return step
        return httpx.Response(200, json=step if step is not None else getattr(self, f"_{role}")(body))

    @staticmethod
    def question(body: dict[str, Any]) -> str:
        for text in _contents(body):
            if match := re.search(r"Customer question: (.*)", text):
                return match.group(1).strip()
        return ""

    def _retrieval(self, body: dict[str, Any]) -> dict[str, Any]:
        if any(m.get("role") == "tool" for m in body["messages"]):
            return completion("Done.")
        return completion(None, tool_calls=[_tool_call("rag_lookup", {"query": self.question(body)})])

    def _lookup(self, body: dict[str, Any]) -> dict[str, Any]:
        order = re.search(r"ORD-[0-9]{4}", self.question(body))
        if order and not any(m.get("role") == "tool" for m in body["messages"]):
            return completion(None, tool_calls=[_tool_call("check_order_status", {"record_id": order.group()})])
        return completion("No order id was given, so no order lookup was possible.")

    def _generator(self, body: dict[str, Any]) -> dict[str, Any]:
        first = re.search(r"^1\. (.*)$", _contents(body)[1], re.M)
        return completion(json.dumps({"quotes": [first.group(1) if first else "?"], "answer": ""}))

    def _composer(self, body: dict[str, Any]) -> dict[str, Any]:
        objs = [o for text in _contents(body) for o in json_objects(text) if "tool" in o]
        rag = next((o for o in objs if o["tool"] == "rag_lookup"), {})
        order = next((o for o in objs if o["tool"] == "check_order_status" and "not_looked_up" not in o), {})
        fields = ("record_id", "status", "order_value_inr", "escalation_score", "escalation_threshold",
                  "escalation_recommended")
        draft = {
            "policy_claims": [c for c in rag.get("claims", []) if c["source_doc_ids"]],
            "policy_error": rag.get("error"),
            "order": {k: order[k] for k in fields} if order and "error" not in order else None,
            "order_error": order.get("error"),
            "needs_order_id": not order and "order" in self.question(body).lower(),
        }
        return completion(json.dumps(draft))

    def _reviewer(self, body: dict[str, Any]) -> dict[str, Any]:
        return completion("ALL CLAIMS SUPPORTED")

    def _editor(self, body: dict[str, Any]) -> dict[str, Any]:
        task = next(t for t in _contents(body) if t.startswith("DRAFT:"))
        draft = re.search(r"DRAFT:\s*(.*?)\s*CONTEXT:", task, re.S).group(1)  # type: ignore[union-attr]
        return completion(json.dumps({"approved": True, "final_answer": draft, "reason": "All supported."}))

    def _unknown(self, body: dict[str, Any]) -> dict[str, Any]:
        return completion("unexpected request")

    def sync_client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def async_client(self) -> httpx.AsyncClient:
        async def ahandler(request: httpx.Request) -> httpx.Response:
            return self.handler(request)

        return httpx.AsyncClient(transport=httpx.MockTransport(ahandler))
