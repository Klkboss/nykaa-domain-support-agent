"""A fake OpenAI-compatible server on httpx.MockTransport: scripted responses, every request recorded."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

BASE_URL = "http://llm.test/v1"  # non-loopback: an unmocked request would hit the network guard


def completion(content: str | None = "ok", usage: tuple[int, int] | None = (11, 3),
               tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
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
