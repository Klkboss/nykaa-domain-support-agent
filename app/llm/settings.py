"""Provider settings, read from the environment when a model client is built (never at import).

MOCK_LLM=true (the default) forces the mock provider whatever LLM_PROVIDER says.
  LLM_PROVIDER       mock (default) | ollama | openai_compatible
  LLM_MODEL          ollama default qwen3:4b-instruct-2507-q4_K_M; required for openai_compatible
  LLM_BASE_URL       required for openai_compatible (an OpenAI-compatible /v1 URL). ${NAME} is replaced by the
                     variable NAME, so an account ID stays in its own variable (Cloudflare:
                     https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/v1)
  LLM_API_KEY        optional; never read from OPENAI_API_KEY, so a key is only sent where it was configured
  LLM_API_KEY_ENV    optional: the NAME of the variable that holds the key (e.g. GROQ_API_KEY), so several provider
                     keys can sit in .env and switching provider changes no secret; it wins over LLM_API_KEY
  LLM_REASONING_EFFORT  optional: none | minimal | low | medium | high, sent as `reasoning_effort` to reasoning
                     models (gpt-oss, Gemini). Unset: not sent. Reasoning text stays in the provider's separate
                     field; only the final answer enters the pipeline.
  OLLAMA_BASE_URL    default http://127.0.0.1:11434 (the /v1 suffix is added)
  LLM_TEMPERATURE    default 0
  LLM_TIMEOUT_S      default 120 (per HTTP attempt)
  LLM_MAX_RETRIES    default 0 for ollama (a local server that is down or too slow will not recover in seconds,
                     and retries would multiply the wait), 2 otherwise (the OpenAI SDK's backoff: 429, 408, 409,
                     5xx, connection errors; honours Retry-After)
  LLM_MAX_TOKENS     default 1024 (completion cap per call; also part of each call's budget reservation)
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from app.core import config
from app.llm.errors import LLMConfigError

Provider = Literal["mock", "ollama", "openai_compatible"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# 127.0.0.1, not localhost: Ollama listens on IPv4 only, and on Windows a synchronous client resolving "localhost"
# tries ::1 first and waits ~2.2 s for the refusal on EVERY new connection (measured; CrewAI opens one per agent
# call), which doubled request latency.
OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"
# Measured on a 6 GB RTX 4050 (evaluation/v2/ollama_measurements.json): fully on GPU, 0 errors in 12 requests, about
# 2.3x faster than qwen2.5:7b-instruct (which spills 18% to CPU there). Owner-approved default.
OLLAMA_DEFAULT_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
# Ollama ignores the key, but the OpenAI SDK needs one; an explicit value also stops it reading OPENAI_API_KEY.
_PLACEHOLDER_KEY = "not-set"


class LLMSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Provider = "mock"
    model: str = ""
    base_url: str = ""
    api_key: SecretStr = SecretStr(_PLACEHOLDER_KEY)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_s: float = Field(default=120.0, gt=0, le=600)
    max_retries: int = Field(default=2, ge=0, le=10)
    max_tokens: int = Field(default=1024, gt=0, le=32_768)
    reasoning_effort: ReasoningEffort | None = None

    @field_validator("base_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return value.rstrip("/")

    @property
    def is_real(self) -> bool:
        return self.provider != "mock"


def load_llm_settings(env: Mapping[str, str] | None = None, mock: bool | None = None) -> LLMSettings:
    """Build validated settings. `mock` defaults to app.core.config.MOCK_LLM (tests patch that constant)."""
    env = os.environ if env is None else env
    mock = config.MOCK_LLM if mock is None else mock
    provider = env.get("LLM_PROVIDER", "mock").strip().lower() or "mock"
    if mock or provider == "mock":
        return LLMSettings()
    if provider not in ("ollama", "openai_compatible"):
        raise LLMConfigError("LLM_PROVIDER must be one of: mock, ollama, openai_compatible.")

    fields: dict[str, object] = {"provider": provider}
    if provider == "ollama":
        fields["base_url"] = env.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_URL).strip().rstrip("/") + "/v1"
        fields["model"] = env.get("LLM_MODEL", "").strip() or OLLAMA_DEFAULT_MODEL
        fields["max_retries"] = 0
    else:
        fields["base_url"] = _expand(env.get("LLM_BASE_URL", "").strip(), env)
        fields["model"] = env.get("LLM_MODEL", "").strip()
        missing = [name for name, key in (("LLM_BASE_URL", "base_url"), ("LLM_MODEL", "model")) if not fields[key]]
        if missing:
            raise LLMConfigError(f"LLM_PROVIDER=openai_compatible needs {' and '.join(missing)}.")
    if env.get("LLM_API_KEY", "").strip():
        fields["api_key"] = env["LLM_API_KEY"].strip()
    if key_name := env.get("LLM_API_KEY_ENV", "").strip():
        if not env.get(key_name, "").strip():  # name the variable, never a value
            raise LLMConfigError(f"LLM_API_KEY_ENV names {key_name}, which is not set.")
        fields["api_key"] = env[key_name].strip()
    if env.get("LLM_REASONING_EFFORT", "").strip():
        fields["reasoning_effort"] = env["LLM_REASONING_EFFORT"].strip().lower()
    for name, key in (("LLM_TEMPERATURE", "temperature"), ("LLM_TIMEOUT_S", "timeout_s"),
                      ("LLM_MAX_RETRIES", "max_retries"), ("LLM_MAX_TOKENS", "max_tokens")):
        if env.get(name, "").strip():
            fields[key] = env[name].strip()
    try:
        return LLMSettings.model_validate(fields)
    except ValidationError as exc:  # name the variable and the rule; never echo the value (it could be a key)
        names = {"base_url": "LLM_BASE_URL/OLLAMA_BASE_URL", "temperature": "LLM_TEMPERATURE",
                 "reasoning_effort": "LLM_REASONING_EFFORT",
                 "timeout_s": "LLM_TIMEOUT_S", "max_retries": "LLM_MAX_RETRIES", "max_tokens": "LLM_MAX_TOKENS"}
        problems = "; ".join(f"{names.get(str(e['loc'][0]), e['loc'][0])}: {e['msg']}" for e in exc.errors())
        raise LLMConfigError(f"Invalid LLM settings: {problems}.") from None


def _expand(url: str, env: Mapping[str, str]) -> str:
    """Replace each ${NAME} in a base URL with that variable; a missing one is named, never guessed."""
    missing = [n for n in _VARIABLE.findall(url) if not env.get(n, "").strip()]
    if missing:
        raise LLMConfigError(f"LLM_BASE_URL uses {', '.join(missing)}, which is not set.")
    return _VARIABLE.sub(lambda m: env[m.group(1)].strip(), url)
