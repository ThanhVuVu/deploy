"""
src/llm/openai_client.py

A thin, retry-aware wrapper around the ``openai`` SDK configured for any
OpenAI-compatible endpoint (NVIDIA NIM, OpenAI, Ollama, …).

Usage
-----
    from src.llm.provider import get_provider
    from src.llm.openai_client import OpenAIClient

    client = OpenAIClient(get_provider())

    response = client.chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "Explain Newton's second law."},
        ]
    )

    print(response.content)        # cleaned text
    print(response.usage)          # token counts (if available)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator, Optional, Sequence

import httpx
from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from .provider import LLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------

class LLMResponse:
    """Lightweight wrapper around a raw :class:`ChatCompletion`."""

    def __init__(self, completion: ChatCompletion) -> None:
        self._completion = completion

    # ------------------------------------------------------------------ #
    # Convenience accessors                                                 #
    # ------------------------------------------------------------------ #

    @property
    def content(self) -> str:
        """Return the first choice's text, stripped of leading/trailing whitespace."""
        choice = self._completion.choices[0]
        return (choice.message.content or "").strip()

    @property
    def usage(self) -> dict[str, int]:
        """Token usage dict: ``{"prompt_tokens": N, "completion_tokens": M, "total_tokens": T}``."""
        u = self._completion.usage
        if u is None:
            return {}
        return {
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        }

    @property
    def model(self) -> str:
        """Model name echoed back by the API."""
        return self._completion.model

    @property
    def finish_reason(self) -> Optional[str]:
        """Stop reason for the first choice (e.g. ``"stop"``, ``"length"``)."""
        return self._completion.choices[0].finish_reason

    @property
    def raw(self) -> ChatCompletion:
        """Unmodified :class:`~openai.types.chat.ChatCompletion` object."""
        return self._completion

    def __str__(self) -> str:  # noqa: D105
        return self.content

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"LLMResponse(model={self.model!r}, "
            f"finish_reason={self.finish_reason!r}, "
            f"content_len={len(self.content)})"
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError)

class OpenAIClient:
    """
    Retry-capable client for any OpenAI-compatible chat-completion endpoint.

    Parameters
    ----------
    provider:
        :class:`~src.llm.provider.LLMProvider` instance produced by
        :func:`~src.llm.provider.get_provider`.
    max_retries:
        Number of automatic retries on transient network/timeout errors (default 3).
    retry_delay:
        Base delay in seconds between retries; doubles each attempt (default 2).
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.provider = provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._client = OpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            http_client=httpx.Client(timeout=provider.timeout),
            max_retries=0,  # We handle retries ourselves for better logging
        )
        logger.debug("OpenAIClient initialised: %r", provider)

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **override_kwargs: Any,
    ) -> LLMResponse:
        """
        Send a chat-completion request and return a :class:`LLMResponse`.

        Parameters
        ----------
        messages:
            List of message dicts (role / content pairs).
        model:
            Override the provider's default model for this call.
        temperature:
            Override sampling temperature for this call.
        max_tokens:
            Override max tokens for this call.
        system_prompt:
            If provided, prepend a ``{"role": "system", …}`` message.
        **override_kwargs:
            Any extra parameters forwarded to the API (e.g. ``top_p``).
        """
        all_messages = list(messages)
        if system_prompt:
            all_messages.insert(0, {"role": "system", "content": system_prompt})

        kwargs = self.provider.chat_kwargs()
        if model:
            kwargs["model"] = model
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs.update(override_kwargs)

        return self._call_with_retry(all_messages, kwargs)

    def chat_simple(self, user_message: str, **kwargs: Any) -> str:
        """
        One-shot helper: send a single user message, return the text response.

        Parameters
        ----------
        user_message:
            Plain text from the user.
        **kwargs:
            Forwarded to :pymeth:`chat`.
        """
        response = self.chat(
            [{"role": "user", "content": user_message}],
            **kwargs,
        )
        return response.content

    def stream(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Yield text delta chunks as they arrive (streaming mode).

        Parameters
        ----------
        messages:
            Conversation history.
        **kwargs:
            Forwarded to :pymeth:`chat`.
        """
        call_kwargs = self.provider.chat_kwargs()
        call_kwargs.update(kwargs)
        call_kwargs["stream"] = True

        logger.debug("Streaming request to model=%r", call_kwargs.get("model"))
        with self._client.chat.completions.create(
            messages=list(messages),  # type: ignore[arg-type]
            **call_kwargs,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _call_with_retry(
        self,
        messages: list[ChatCompletionMessageParam],
        kwargs: dict[str, Any],
    ) -> LLMResponse:
        """Attempt the API call up to ``max_retries + 1`` times."""
        last_exc: Exception | None = None
        delay = self.retry_delay

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    "Chat completion attempt %d/%d  model=%r",
                    attempt + 1,
                    self.max_retries + 1,
                    kwargs.get("model"),
                )
                completion: ChatCompletion = self._client.chat.completions.create(
                    messages=messages,  # type: ignore[arg-type]
                    **kwargs,
                )
                logger.debug(
                    "Chat completion succeeded  finish_reason=%r",
                    completion.choices[0].finish_reason,
                )
                return LLMResponse(completion)

            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "Transient error (%s); retrying in %.1fs …",
                        type(exc).__name__,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2  # exponential back-off

            except APIStatusError as exc:
                # 429 rate-limit → also retry; other 4xx/5xx → re-raise immediately
                if exc.status_code == 429 and attempt < self.max_retries:
                    last_exc = exc
                    retry_after = float(
                        exc.response.headers.get("retry-after", delay)
                    )
                    logger.warning(
                        "Rate-limited (429); retrying in %.1fs …", retry_after
                    )
                    time.sleep(retry_after)
                    delay *= 2
                else:
                    logger.error("API error %d: %s", exc.status_code, exc.message)
                    raise

        raise RuntimeError(
            f"All {self.max_retries + 1} attempts failed."
        ) from last_exc

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"OpenAIClient(provider={self.provider!r}, "
            f"max_retries={self.max_retries})"
        )
