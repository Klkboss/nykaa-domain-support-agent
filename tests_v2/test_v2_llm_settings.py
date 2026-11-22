"""v2 Stage A: provider settings. With nothing set the provider is mock; MOCK_LLM=true always wins."""

from __future__ import annotations

import pytest

from app.llm.errors import LLMConfigError
from app.llm.settings import OLLAMA_DEFAULT_MODEL, load_llm_settings


def test_no_environment_means_mock():
    assert load_llm_settings({}, mock=False).provider == "mock"
    assert not load_llm_settings({}, mock=False).is_real


def test_mock_llm_true_forces_mock_whatever_the_provider_says():
    s = load_llm_settings({"LLM_PROVIDER": "ollama", "LLM_MODEL": "x"}, mock=True)
    assert s.provider == "mock"


def test_default_follows_the_capstone_switch(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config, "MOCK_LLM", True)
    assert load_llm_settings({"LLM_PROVIDER": "ollama"}).provider == "mock"


def test_ollama_defaults():
    s = load_llm_settings({"LLM_PROVIDER": "ollama"}, mock=False)
    assert (s.provider, s.model, s.base_url) == ("ollama", OLLAMA_DEFAULT_MODEL, "http://127.0.0.1:11434/v1")
    assert (s.temperature, s.max_retries) == (0.0, 0)  # local: failures must not multiply


def test_ollama_overrides():
    s = load_llm_settings({"LLM_PROVIDER": "Ollama", "OLLAMA_BASE_URL": "http://gpu-box:11434/",
                           "LLM_MODEL": "qwen2.5:3b-instruct", "LLM_TEMPERATURE": "0.2", "LLM_TIMEOUT_S": "30",
                           "LLM_MAX_RETRIES": "1", "LLM_MAX_TOKENS": "512"}, mock=False)
    assert (s.base_url, s.model, s.temperature, s.timeout_s, s.max_retries, s.max_tokens) == (
        "http://gpu-box:11434/v1", "qwen2.5:3b-instruct", 0.2, 30.0, 1, 512)


def test_openai_compatible_needs_base_url_and_model():
    with pytest.raises(LLMConfigError, match="LLM_BASE_URL and LLM_MODEL"):
        load_llm_settings({"LLM_PROVIDER": "openai_compatible"}, mock=False)
    s = load_llm_settings({"LLM_PROVIDER": "openai_compatible", "LLM_BASE_URL": "https://api.example.com/v1/",
                           "LLM_MODEL": "some-model"}, mock=False)
    assert (s.base_url, s.model) == ("https://api.example.com/v1", "some-model")


def test_unknown_provider_is_a_clear_error():
    with pytest.raises(LLMConfigError, match="mock, ollama, openai_compatible"):
        load_llm_settings({"LLM_PROVIDER": "groq"}, mock=False)


@pytest.mark.parametrize("name, value", [("LLM_TEMPERATURE", "hot"), ("LLM_TEMPERATURE", "3"),
                                         ("LLM_TIMEOUT_S", "-5"), ("LLM_MAX_RETRIES", "-1"),
                                         ("LLM_MAX_TOKENS", "lots"), ("OLLAMA_BASE_URL", "gpu-box:11434")])
def test_invalid_values_name_the_variable_but_never_echo_the_value(name, value):
    with pytest.raises(LLMConfigError) as err:
        load_llm_settings({"LLM_PROVIDER": "ollama", name: value}, mock=False)
    assert name in str(err.value) and value not in str(err.value)


def test_api_key_is_never_shown():
    s = load_llm_settings({"LLM_PROVIDER": "openai_compatible", "LLM_BASE_URL": "https://api.example.com/v1",
                           "LLM_MODEL": "m", "LLM_API_KEY": "sk-secret-123"}, mock=False)
    assert "sk-secret-123" not in repr(s) and "sk-secret-123" not in str(s.model_dump())
    assert s.api_key.get_secret_value() == "sk-secret-123"


def test_openai_api_key_in_the_environment_is_not_picked_up():
    s = load_llm_settings({"LLM_PROVIDER": "ollama", "OPENAI_API_KEY": "sk-openai"}, mock=False)
    assert s.api_key.get_secret_value() != "sk-openai"


HOSTED = {"LLM_PROVIDER": "openai_compatible", "LLM_MODEL": "openai/gpt-oss-120b"}


def test_the_key_can_be_named_by_its_variable_and_is_never_shown():
    s = load_llm_settings({**HOSTED, "LLM_BASE_URL": "https://api.groq.com/openai/v1",
                           "LLM_API_KEY_ENV": "GROQ_API_KEY", "GROQ_API_KEY": "gsk-secret-1",
                           "LLM_API_KEY": "other"}, mock=False)
    assert s.api_key.get_secret_value() == "gsk-secret-1"  # the named variable wins
    assert "gsk-secret-1" not in repr(s) and "gsk-secret-1" not in str(s.model_dump())


def test_a_named_key_variable_that_is_unset_is_a_clear_error():
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY, which is not set"):
        load_llm_settings({**HOSTED, "LLM_BASE_URL": "https://api.groq.com/openai/v1",
                           "LLM_API_KEY_ENV": "GROQ_API_KEY"}, mock=False)


def test_the_base_url_takes_variables_so_an_account_id_is_never_hard_coded():
    url = "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/v1"
    s = load_llm_settings({**HOSTED, "LLM_BASE_URL": url, "CLOUDFLARE_ACCOUNT_ID": "abc123"}, mock=False)
    assert s.base_url == "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1"
    with pytest.raises(LLMConfigError, match="CLOUDFLARE_ACCOUNT_ID, which is not set"):
        load_llm_settings({**HOSTED, "LLM_BASE_URL": url}, mock=False)


def test_reasoning_effort_is_optional_and_validated():
    base = {**HOSTED, "LLM_BASE_URL": "https://api.groq.com/openai/v1"}
    assert load_llm_settings(base, mock=False).reasoning_effort is None
    assert load_llm_settings({**base, "LLM_REASONING_EFFORT": "Low"}, mock=False).reasoning_effort == "low"
    with pytest.raises(LLMConfigError, match="LLM_REASONING_EFFORT"):
        load_llm_settings({**base, "LLM_REASONING_EFFORT": "max"}, mock=False)
