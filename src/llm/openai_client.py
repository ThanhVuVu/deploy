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
import re
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
        max_retries: int = 5,
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
        self._normalize_model_parameters(kwargs)

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
        self._normalize_model_parameters(call_kwargs)
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
                if self._try_repair_bad_request(exc, kwargs):
                    continue
                # Retry on rate-limit (429) or transient server errors (500, 502, 503, 504)
                retryable_status_codes = {429, 500, 502, 503, 504}
                if exc.status_code in retryable_status_codes and attempt < self.max_retries:
                    last_exc = exc
                    # Use Retry-After header if present, otherwise use exponential back-off
                    retry_after = delay
                    header_retry_after = exc.response.headers.get("retry-after")
                    if header_retry_after:
                        try:
                            retry_after = float(header_retry_after)
                        except ValueError:
                            pass
                    
                    logger.warning(
                        "API error %d; retrying in %.1fs … (%s)",
                        exc.status_code,
                        retry_after,
                        exc.message
                    )
                    time.sleep(retry_after)
                    delay *= 2
                else:
                    logger.error("API error %d: %s", exc.status_code, exc.message)
                    raise

        raise RuntimeError(
            f"All {self.max_retries + 1} attempts failed."
        ) from last_exc

    def _normalize_model_parameters(self, kwargs: dict[str, Any]) -> None:
        """Normalize request kwargs for model-specific Chat Completions rules."""
        self._normalize_token_parameter(kwargs)
        self._normalize_sampling_parameters(kwargs)

    def _normalize_token_parameter(self, kwargs: dict[str, Any]) -> None:
        """
        Ensure token limit parameter matches model requirements.

        GPT-5 class models require ``max_completion_tokens`` instead of
        ``max_tokens``. For all other models we keep existing behavior.
        """
        if "max_completion_tokens" in kwargs:
            kwargs.pop("max_tokens", None)
            return

        model_name = str(kwargs.get("model", "")).lower()
        if self._requires_max_completion_tokens(model_name) and "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

    def _normalize_sampling_parameters(self, kwargs: dict[str, Any]) -> None:
        """Remove sampling controls rejected by GPT-5/o-series chat models."""
        model_name = str(kwargs.get("model", "")).lower()
        if not self._requires_default_sampling(model_name):
            return

        for param in self._default_only_sampling_params():
            kwargs.pop(param, None)

    def _try_repair_bad_request(
        self,
        exc: APIStatusError,
        kwargs: dict[str, Any],
    ) -> bool:
        """Adjust known model-compatibility request issues and retry once."""
        if self._try_convert_max_tokens(exc, kwargs):
            return True
        if self._try_remove_unsupported_parameter(exc, kwargs):
            return True
        if self._try_remove_unsupported_value(exc, kwargs):
            return True
        return False

    def _try_convert_max_tokens(
        self,
        exc: APIStatusError,
        kwargs: dict[str, Any],
    ) -> bool:
        """
        Convert ``max_tokens`` to ``max_completion_tokens`` on known 400 errors.

        Returns ``True`` if kwargs were adjusted and call should be retried.
        """
        if exc.status_code != 400:
            return False

        message = (exc.message or "").lower()
        if "max_tokens" not in message or "max_completion_tokens" not in message:
            return False
        if "max_tokens" not in kwargs:
            return False

        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        logger.warning(
            "Converted max_tokens to max_completion_tokens after API 400 for model=%r",
            kwargs.get("model"),
        )
        return True

    def _try_remove_unsupported_parameter(
        self,
        exc: APIStatusError,
        kwargs: dict[str, Any],
    ) -> bool:
        """Drop a request parameter when the API explicitly says it is unsupported."""
        if exc.status_code != 400:
            return False

        if "unsupported parameter" not in self._error_message(exc):
            return False

        param = self._error_param(exc)
        if not param or param == "max_tokens" or param not in kwargs:
            return False

        kwargs.pop(param, None)
        logger.warning(
            "Removed unsupported parameter %r after API 400 for model=%r",
            param,
            kwargs.get("model"),
        )
        return True

    def _try_remove_unsupported_value(
        self,
        exc: APIStatusError,
        kwargs: dict[str, Any],
    ) -> bool:
        """Drop a parameter when the API says only the default value is allowed."""
        if exc.status_code != 400:
            return False

        message = self._error_message(exc)
        if "unsupported value" not in message and "only the default" not in message:
            return False

        param = self._error_param(exc)
        if not param or param not in kwargs:
            return False

        kwargs.pop(param, None)
        logger.warning(
            "Removed unsupported value for parameter %r after API 400 for model=%r",
            param,
            kwargs.get("model"),
        )
        return True

    @staticmethod
    def _error_message(exc: APIStatusError) -> str:
        return (getattr(exc, "message", "") or str(exc) or "").lower()

    @classmethod
    def _error_param(cls, exc: APIStatusError) -> str:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            raw_param = body.get("param")
            if isinstance(raw_param, str) and raw_param:
                return raw_param

            error = body.get("error")
            if isinstance(error, dict):
                raw_param = error.get("param")
                if isinstance(raw_param, str) and raw_param:
                    return raw_param

        message = cls._error_message(exc)
        match = re.search(r"param(?:eter)?[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']", message)
        if match:
            return match.group(1)
        match = re.search(r"unsupported (?:parameter|value): [\"']([^\"']+)[\"']", message)
        if match:
            return match.group(1)
        match = re.search(r"[\"']([a-z_]+)[\"'] does not support", message)
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def _requires_max_completion_tokens(model_name: str) -> bool:
        markers = ("gpt-5", "o1", "o3", "o4")
        return any(marker in model_name for marker in markers)

    @staticmethod
    def _requires_default_sampling(model_name: str) -> bool:
        markers = ("gpt-5", "o1", "o3", "o4")
        return any(marker in model_name for marker in markers)

    @staticmethod
    def _default_only_sampling_params() -> tuple[str, ...]:
        return (
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"OpenAIClient(provider={self.provider!r}, "
            f"max_retries={self.max_retries})"
        )
