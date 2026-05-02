"""
src/llm/provider.py

Centralised LLM provider configuration.

Usage
-----
    from src.llm.provider import get_provider, LLMProvider

    provider = get_provider()          # reads from env / .env file
    # or override per-call:
    provider = get_provider(model="nvidia/llama-3.1-nemotron-70b-instruct")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Supported provider back-ends
# ---------------------------------------------------------------------------

class ProviderBackend(str, Enum):
    """Supported API back-ends (all OpenAI-compatible)."""
    NVIDIA = "nvidia"
    OPENAI = "openai"
    OLLAMA = "ollama"


# ---------------------------------------------------------------------------
# Default models per back-end
# ---------------------------------------------------------------------------

_DEFAULT_MODELS: dict[ProviderBackend, str] = {
    ProviderBackend.NVIDIA: "nvidia/llama-3.1-nemotron-70b-instruct",
    ProviderBackend.OPENAI: "gpt-4o",
    ProviderBackend.OLLAMA: "llama3",
}

_DEFAULT_BASE_URLS: dict[ProviderBackend, str] = {
    ProviderBackend.NVIDIA: "https://integrate.api.nvidia.com/v1/",
    ProviderBackend.OPENAI: "https://api.openai.com/v1/",
    ProviderBackend.OLLAMA: "http://localhost:11434/v1/",
}

_DEFAULT_ONLY_SAMPLING_PARAMS = (
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMProvider:
    """Immutable snapshot of a provider's connection configuration."""

    backend: ProviderBackend
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: float = 300.0
    extra_kwargs: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Derived helpers                                                       #
    # ------------------------------------------------------------------ #

    @property
    def is_thinking_model(self) -> bool:
        """Return True if the model supports extended reasoning / thinking."""
        _thinking_keywords = ("qwen3", "deepseek-r1", "nemotron", "o1", "o3")
        return any(kw in self.model.lower() for kw in _thinking_keywords)

    @property
    def uses_max_completion_tokens(self) -> bool:
        """Return True when Chat Completions requires max_completion_tokens."""
        model_name = self.model.lower()
        openai_completion_token_models = ("gpt-5", "o1", "o3", "o4")
        return self.backend == ProviderBackend.OPENAI and any(
            marker in model_name for marker in openai_completion_token_models
        )

    @property
    def uses_default_sampling_params(self) -> bool:
        """Return True when the model rejects custom sampling controls."""
        model_name = self.model.lower()
        default_sampling_models = ("gpt-5", "o1", "o3", "o4")
        return self.backend == ProviderBackend.OPENAI and any(
            marker in model_name for marker in default_sampling_models
        )

    def chat_kwargs(self) -> dict:
        """Build kwargs ready to unpack into ``client.chat.completions.create``."""
        kwargs: dict = {
            "model": self.model,
            **self.extra_kwargs,
        }
        if not self.uses_default_sampling_params:
            kwargs.setdefault("temperature", self.temperature)
        else:
            for param in _DEFAULT_ONLY_SAMPLING_PARAMS:
                kwargs.pop(param, None)

        token_key = (
            "max_completion_tokens"
            if self.uses_max_completion_tokens
            else "max_tokens"
        )
        kwargs.setdefault(token_key, self.max_tokens)
        # Some NVIDIA thinking models require this flag
        if self.is_thinking_model and self.backend == ProviderBackend.NVIDIA:
            kwargs.setdefault("chat_template_kwargs", {"thinking": {"type": "disabled"}})
        return kwargs

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"LLMProvider(backend={self.backend.value!r}, model={self.model!r}, "
            f"base_url={self.base_url!r})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(
    *,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: float = 120.0,
    **extra_kwargs,
) -> LLMProvider:
    """
    Build an :class:`LLMProvider` from parameters and/or environment variables.

    Resolution order (highest wins):
        1. Explicit argument passed to this function
        2. Environment variable (``NVIDIA_API_KEY``, ``OPENAI_API_KEY``, …)
        3. Hard-coded default for the resolved back-end

    Parameters
    ----------
    backend:
        One of ``"nvidia"`` | ``"openai"`` | ``"ollama"``.
        Falls back to ``PROVIDER_BACKEND`` env var, then ``"nvidia"``.
    model:
        Model name/slug.  Falls back to ``LLM_MODEL`` env var.
    api_key:
        API secret key.  Falls back to ``NVIDIA_API_KEY`` / ``OPENAI_API_KEY``.
    base_url:
        API base URL.  Falls back to ``NVIDIA_BASE_URL`` / ``OPENAI_BASE_URL``.
    temperature:
        Sampling temperature (default ``0.2``).
    max_tokens:
        Maximum completion tokens (default ``4096``).
    timeout:
        HTTP request timeout in seconds (default ``120``).
    **extra_kwargs:
        Forwarded verbatim to :pymeth:`LLMProvider.chat_kwargs`.
    """
    # ---- resolve backend ---------------------------------------------------
    raw_backend = (
        backend
        or os.getenv("PROVIDER_BACKEND", "nvidia")
    ).lower()
    try:
        resolved_backend = ProviderBackend(raw_backend)
    except ValueError:
        supported = [b.value for b in ProviderBackend]
        raise ValueError(
            f"Unknown backend {raw_backend!r}. Supported: {supported}"
        ) from None

    # ---- resolve model -----------------------------------------------------
    resolved_model = (
        model
        or os.getenv("LLM_MODEL")
        or _DEFAULT_MODELS[resolved_backend]
    )

    # ---- resolve api_key ---------------------------------------------------
    _key_env_map = {
        ProviderBackend.NVIDIA: "NVIDIA_API_KEY",
        ProviderBackend.OPENAI: "OPENAI_API_KEY",
        ProviderBackend.OLLAMA: "OLLAMA_API_KEY",
    }
    resolved_key = (
        api_key
        or os.getenv(_key_env_map[resolved_backend], "")
    )
    if not resolved_key and resolved_backend != ProviderBackend.OLLAMA:
        raise EnvironmentError(
            f"No API key found for backend {resolved_backend.value!r}. "
            f"Set the {_key_env_map[resolved_backend]!r} environment variable "
            "or pass `api_key=...` explicitly."
        )

    # ---- resolve base_url --------------------------------------------------
    _url_env_map = {
        ProviderBackend.NVIDIA: "NVIDIA_BASE_URL",
        ProviderBackend.OPENAI: "OPENAI_BASE_URL",
        ProviderBackend.OLLAMA: "OLLAMA_BASE_URL",
    }
    resolved_url = (
        base_url
        or os.getenv(_url_env_map[resolved_backend])
        or _DEFAULT_BASE_URLS[resolved_backend]
    )

    return LLMProvider(
        backend=resolved_backend,
        model=resolved_model,
        api_key=resolved_key,
        base_url=resolved_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_kwargs=extra_kwargs,
    )
