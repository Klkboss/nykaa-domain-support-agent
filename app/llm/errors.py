"""Typed, user-safe provider errors.

Messages say what to check; they never echo provider response bodies, prompts or keys. `classify()` maps an
exception (or anything in its cause/context chain, because CrewAI and AutoGen may wrap it) to one of these.
"""

from __future__ import annotations

import httpx
import openai


class LLMError(RuntimeError):
    kind = "llm_error"


class LLMConfigError(LLMError):
    kind = "config"


class LLMUnavailable(LLMError):
    kind = "unavailable"


class LLMTimeout(LLMError):
    kind = "timeout"


class LLMRateLimited(LLMError):
    kind = "rate_limited"


class LLMAuthError(LLMError):
    kind = "auth"


class LLMModelNotFound(LLMError):
    kind = "model_not_found"


class LLMResponseError(LLMError):
    kind = "bad_response"


def _map(exc: BaseException) -> LLMError | None:
    # APITimeoutError subclasses APIConnectionError: check it first.
    if isinstance(exc, openai.APITimeoutError | httpx.TimeoutException):
        return LLMTimeout("The model server did not answer in time (LLM_TIMEOUT_S), after retries.")
    if isinstance(exc, openai.APIConnectionError | httpx.TransportError):
        return LLMUnavailable("The model server could not be reached. Check LLM_BASE_URL / OLLAMA_BASE_URL and "
                              "that the server is running.")
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimited("The model server rate-limited the request (HTTP 429), after retries.")
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return LLMAuthError("The model server rejected the credentials. Check LLM_API_KEY.")
    if isinstance(exc, openai.NotFoundError):
        return LLMModelNotFound("The model server does not have this model. Check LLM_MODEL "
                                "(for Ollama: `ollama pull <model>`).")
    if isinstance(exc, openai.APIStatusError):
        return LLMResponseError(f"The model server returned HTTP {exc.status_code}.")
    return None


def classify(exc: BaseException) -> LLMError | None:
    """The provider error behind `exc`, or None when it is not a provider error."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, LLMError):
            return current
        mapped = _map(current)
        if mapped is not None:
            return mapped
        current = current.__cause__ or current.__context__
    return None
