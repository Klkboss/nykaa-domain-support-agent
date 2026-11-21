"""Provider settings, read from the environment when a model client is built (never at import).

MOCK_LLM=true (the default) forces the mock provider whatever LLM_PROVIDER says.
  LLM_PROVIDER       mock (default) | ollama | openai_compatible
  LLM_MODEL          ollama default qwen2.5:7b-instruct; required for openai_compatible
  LLM_BASE_URL       required for openai_compatible (an OpenAI-compatible /v1 URL)
  LLM_API_KEY        optional; never read from OPENAI_API_KEY, so a key is only sent where it was configured
  OLLAMA_BASE_URL    default http://localhost:11434 (the /v1 suffix is added)
  LLM_TEMPERATURE    default 0
  LLM_TIMEOUT_S      default 120 (per HTTP attempt)
  LLM_MAX_RETRIES    default 2 (the OpenAI SDK's backoff: 429, 408, 409, 5xx, connection errors; honours Retry-After)
  LLM_MAX_TOKENS     default 1024 (completion cap per call; also part of each call's budget reservation)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from app.core import config
from app.llm.errors import LLMConfigError

Provider = Literal["mock", "ollama", "openai_compatible"]
OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "qwen2.5:7b-instruct"
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
    else:
        fields["base_url"] = env.get("LLM_BASE_URL", "").strip()
        fields["model"] = env.get("LLM_MODEL", "").strip()
        missing = [name for name, key in (("LLM_BASE_URL", "base_url"), ("LLM_MODEL", "model")) if not fields[key]]
        if missing:
            raise LLMConfigError(f"LLM_PROVIDER=openai_compatible needs {' and '.join(missing)}.")
    if env.get("LLM_API_KEY", "").strip():
        fields["api_key"] = env["LLM_API_KEY"].strip()
    for name, key in (("LLM_TEMPERATURE", "temperature"), ("LLM_TIMEOUT_S", "timeout_s"),
                      ("LLM_MAX_RETRIES", "max_retries"), ("LLM_MAX_TOKENS", "max_tokens")):
        if env.get(name, "").strip():
            fields[key] = env[name].strip()
    try:
        return LLMSettings.model_validate(fields)
    except ValidationError as exc:  # name the variable and the rule; never echo the value (it could be a key)
        names = {"base_url": "LLM_BASE_URL/OLLAMA_BASE_URL", "temperature": "LLM_TEMPERATURE",
                 "timeout_s": "LLM_TIMEOUT_S", "max_retries": "LLM_MAX_RETRIES", "max_tokens": "LLM_MAX_TOKENS"}
        problems = "; ".join(f"{names.get(str(e['loc'][0]), e['loc'][0])}: {e['msg']}" for e in exc.errors())
        raise LLMConfigError(f"Invalid LLM settings: {problems}.") from None
