"""
tests/test_llm_provider.py

Unit tests for src/llm/provider.py

Run with:
    pytest tests/test_llm_provider.py -v
"""

import os
import pytest

from src.llm.provider import (
    LLMProvider,
    ProviderBackend,
    get_provider,
    _DEFAULT_MODELS,
    _DEFAULT_BASE_URLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def clean_env(monkeypatch):
    """Remove all LLM-related env vars so tests start from a clean state."""
    for key in (
        "NVIDIA_API_KEY", "NVIDIA_BASE_URL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "OLLAMA_API_KEY", "OLLAMA_BASE_URL",
        "PROVIDER_BACKEND", "LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


# ---------------------------------------------------------------------------
# get_provider — backend resolution
# ---------------------------------------------------------------------------

class TestGetProviderBackend:
    def test_explicit_backend_nvidia(self, clean_env):
        p = get_provider(backend="nvidia", api_key="key-123")
        assert p.backend == ProviderBackend.NVIDIA

    def test_explicit_backend_openai(self, clean_env):
        p = get_provider(backend="openai", api_key="key-abc")
        assert p.backend == ProviderBackend.OPENAI

    def test_explicit_backend_ollama(self, clean_env):
        p = get_provider(backend="ollama")
        assert p.backend == ProviderBackend.OLLAMA

    def test_env_backend_override(self, clean_env, monkeypatch):
        monkeypatch.setenv("PROVIDER_BACKEND", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        p = get_provider()
        assert p.backend == ProviderBackend.OPENAI

    def test_default_backend_is_nvidia(self, clean_env, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
        p = get_provider()
        assert p.backend == ProviderBackend.NVIDIA

    def test_unknown_backend_raises(self, clean_env):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_provider(backend="gemini", api_key="x")

    def test_backend_case_insensitive(self, clean_env):
        p = get_provider(backend="OPENAI", api_key="key")
        assert p.backend == ProviderBackend.OPENAI


# ---------------------------------------------------------------------------
# get_provider — model resolution
# ---------------------------------------------------------------------------

class TestGetProviderModel:
    def test_explicit_model(self, clean_env):
        p = get_provider(backend="openai", api_key="k", model="gpt-4-turbo")
        assert p.model == "gpt-4-turbo"

    def test_env_model_override(self, clean_env, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "env-model-slug")
        monkeypatch.setenv("NVIDIA_API_KEY", "k")
        p = get_provider()
        assert p.model == "env-model-slug"

    def test_default_model_nvidia(self, clean_env, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "k")
        p = get_provider()
        assert p.model == _DEFAULT_MODELS[ProviderBackend.NVIDIA]

    def test_default_model_openai(self, clean_env):
        p = get_provider(backend="openai", api_key="k")
        assert p.model == _DEFAULT_MODELS[ProviderBackend.OPENAI]


# ---------------------------------------------------------------------------
# get_provider — API key resolution
# ---------------------------------------------------------------------------

class TestGetProviderApiKey:
    def test_explicit_api_key(self, clean_env):
        p = get_provider(backend="openai", api_key="explicit-key")
        assert p.api_key == "explicit-key"

    def test_env_api_key_nvidia(self, clean_env, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "env-nv-key")
        p = get_provider()
        assert p.api_key == "env-nv-key"

    def test_env_api_key_openai(self, clean_env, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-oai-key")
        p = get_provider(backend="openai")
        assert p.api_key == "env-oai-key"

    def test_missing_api_key_raises(self, clean_env):
        with pytest.raises(EnvironmentError, match="No API key found"):
            get_provider(backend="nvidia")

    def test_ollama_no_key_required(self, clean_env):
        """Ollama is local — no API key should be needed."""
        p = get_provider(backend="ollama")
        assert p.api_key == ""


# ---------------------------------------------------------------------------
# get_provider — base URL resolution
# ---------------------------------------------------------------------------

class TestGetProviderBaseUrl:
    def test_explicit_base_url(self, clean_env):
        p = get_provider(backend="openai", api_key="k", base_url="http://custom/v1/")
        assert p.base_url == "http://custom/v1/"

    def test_env_base_url_nvidia(self, clean_env, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "k")
        monkeypatch.setenv("NVIDIA_BASE_URL", "http://env-nvidia/v1/")
        p = get_provider()
        assert p.base_url == "http://env-nvidia/v1/"

    def test_default_base_url_nvidia(self, clean_env, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "k")
        p = get_provider()
        assert p.base_url == _DEFAULT_BASE_URLS[ProviderBackend.NVIDIA]


# ---------------------------------------------------------------------------
# get_provider — scalar params
# ---------------------------------------------------------------------------

class TestGetProviderScalars:
    def test_temperature(self, clean_env):
        p = get_provider(backend="openai", api_key="k", temperature=0.9)
        assert p.temperature == 0.9

    def test_max_tokens(self, clean_env):
        p = get_provider(backend="openai", api_key="k", max_tokens=1024)
        assert p.max_tokens == 1024

    def test_timeout(self, clean_env):
        p = get_provider(backend="openai", api_key="k", timeout=60.0)
        assert p.timeout == 60.0


# ---------------------------------------------------------------------------
# LLMProvider — is_thinking_model
# ---------------------------------------------------------------------------

class TestIsThinkingModel:
    @pytest.mark.parametrize("model,expected", [
        ("nvidia/llama-3.1-nemotron-70b-instruct", True),
        ("qwen/qwen3.5-122b", True),
        ("deepseek-r1-distill-llama-70b", True),
        ("openai/o1-preview", True),
        ("gpt-4o", False),
        ("mistralai/mixtral-8x7b", False),
    ])
    def test_thinking_detection(self, model, expected):
        p = LLMProvider(
            backend=ProviderBackend.NVIDIA,
            model=model,
            api_key="k",
            base_url="http://x/",
        )
        assert p.is_thinking_model is expected


# ---------------------------------------------------------------------------
# LLMProvider — chat_kwargs
# ---------------------------------------------------------------------------

class TestChatKwargs:
    def test_basic_keys_present(self):
        p = LLMProvider(
            backend=ProviderBackend.OPENAI,
            model="gpt-4o",
            api_key="k",
            base_url="http://x/",
            temperature=0.5,
            max_tokens=512,
        )
        kw = p.chat_kwargs()
        assert kw["model"] == "gpt-4o"
        assert kw["temperature"] == 0.5
        assert kw["max_tokens"] == 512

    def test_gpt5_uses_max_completion_tokens(self):
        p = LLMProvider(
            backend=ProviderBackend.OPENAI,
            model="gpt-5.5",
            api_key="k",
            base_url="http://x/",
            max_tokens=12000,
        )
        kw = p.chat_kwargs()
        assert kw["model"] == "gpt-5.5"
        assert kw["max_completion_tokens"] == 12000
        assert "temperature" not in kw
        assert "max_tokens" not in kw

    def test_prefixed_gpt5_uses_max_completion_tokens(self):
        p = LLMProvider(
            backend=ProviderBackend.OPENAI,
            model="openai/gpt-5.5",
            api_key="k",
            base_url="http://x/",
            max_tokens=12000,
            extra_kwargs={"top_p": 0.9},
        )
        kw = p.chat_kwargs()
        assert kw["max_completion_tokens"] == 12000
        assert "top_p" not in kw
        assert "temperature" not in kw
        assert "max_tokens" not in kw

    def test_thinking_flag_injected_for_nvidia(self):
        p = LLMProvider(
            backend=ProviderBackend.NVIDIA,
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            api_key="k",
            base_url="http://x/",
        )
        kw = p.chat_kwargs()
        assert "chat_template_kwargs" in kw
        assert kw["chat_template_kwargs"]["thinking"]["type"] == "disabled"

    def test_thinking_flag_not_injected_for_openai(self):
        p = LLMProvider(
            backend=ProviderBackend.OPENAI,
            model="openai/o1-preview",
            api_key="k",
            base_url="http://x/",
        )
        kw = p.chat_kwargs()
        assert "chat_template_kwargs" not in kw

    def test_extra_kwargs_forwarded(self):
        p = LLMProvider(
            backend=ProviderBackend.OPENAI,
            model="gpt-4o",
            api_key="k",
            base_url="http://x/",
            extra_kwargs={"top_p": 0.95},
        )
        kw = p.chat_kwargs()
        assert kw["top_p"] == 0.95

    def test_repr(self):
        p = LLMProvider(
            backend=ProviderBackend.NVIDIA,
            model="some-model",
            api_key="k",
            base_url="http://x/",
        )
        r = repr(p)
        assert "nvidia" in r
        assert "some-model" in r
