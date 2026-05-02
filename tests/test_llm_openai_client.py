"""
tests/test_llm_openai_client.py

Unit tests for src/llm/openai_client.py  (all API calls are mocked — no
real HTTP requests are made)

Run with:
    pytest tests/test_llm_openai_client.py -v
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest
from openai import APIConnectionError, APITimeoutError, APIStatusError

from src.llm.provider import LLMProvider, ProviderBackend
from src.llm.openai_client import OpenAIClient, LLMResponse


# ---------------------------------------------------------------------------
# Helpers — build fake OpenAI ChatCompletion objects
# ---------------------------------------------------------------------------

def _make_completion(
    content: str = "Hello!",
    model: str = "test-model",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Return a MagicMock that mimics openai.types.chat.ChatCompletion."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    completion = MagicMock()
    completion.choices = [choice]
    completion.model = model
    completion.usage = usage
    return completion


def _make_provider(**kwargs) -> LLMProvider:
    defaults = dict(
        backend=ProviderBackend.OPENAI,
        model="gpt-4o",
        api_key="test-key",
        base_url="http://fake-api/v1/",
        temperature=0.2,
        max_tokens=256,
        timeout=10.0,
    )
    defaults.update(kwargs)
    return LLMProvider(**defaults)


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------

class TestLLMResponse:
    def test_content_stripped(self):
        c = _make_completion(content="  Hello!  ")
        r = LLMResponse(c)
        assert r.content == "Hello!"

    def test_usage_dict(self):
        c = _make_completion(prompt_tokens=8, completion_tokens=4)
        r = LLMResponse(c)
        assert r.usage == {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}

    def test_usage_empty_when_none(self):
        c = _make_completion()
        c.usage = None
        r = LLMResponse(c)
        assert r.usage == {}

    def test_model(self):
        c = _make_completion(model="my-model")
        assert LLMResponse(c).model == "my-model"

    def test_finish_reason(self):
        c = _make_completion(finish_reason="length")
        assert LLMResponse(c).finish_reason == "length"

    def test_raw_is_original(self):
        c = _make_completion()
        r = LLMResponse(c)
        assert r.raw is c

    def test_str(self):
        c = _make_completion(content="The answer is 42.")
        assert str(LLMResponse(c)) == "The answer is 42."

    def test_repr(self):
        c = _make_completion(model="x", finish_reason="stop", content="hi")
        assert "LLMResponse" in repr(LLMResponse(c))


# ---------------------------------------------------------------------------
# OpenAIClient — happy path
# ---------------------------------------------------------------------------

class TestOpenAIClientChat:
    @patch("src.llm.openai_client.OpenAI")
    def test_chat_returns_response(self, MockOpenAI):
        completion = _make_completion("The sky is blue.")
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = completion

        client = OpenAIClient(_make_provider())
        resp = client.chat([{"role": "user", "content": "What colour is the sky?"}])

        assert isinstance(resp, LLMResponse)
        assert resp.content == "The sky is blue."

    @patch("src.llm.openai_client.OpenAI")
    def test_chat_forwards_messages(self, MockOpenAI):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = _make_completion()

        client = OpenAIClient(_make_provider())
        msgs = [{"role": "user", "content": "Hello"}]
        client.chat(msgs)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["messages"] == msgs

    @patch("src.llm.openai_client.OpenAI")
    def test_chat_system_prompt_prepended(self, MockOpenAI):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = _make_completion()

        client = OpenAIClient(_make_provider())
        client.chat(
            [{"role": "user", "content": "Hi"}],
            system_prompt="You are a helpful tutor.",
        )

        sent_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert sent_messages[0] == {"role": "system", "content": "You are a helpful tutor."}
        assert sent_messages[1] == {"role": "user", "content": "Hi"}

    @patch("src.llm.openai_client.OpenAI")
    def test_chat_per_call_overrides(self, MockOpenAI):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = _make_completion()

        client = OpenAIClient(_make_provider(temperature=0.2, max_tokens=256))
        client.chat(
            [{"role": "user", "content": "hi"}],
            temperature=0.9,
            max_tokens=128,
            model="gpt-4-turbo",
        )

        kw = mock_client.chat.completions.create.call_args.kwargs
        assert kw["temperature"] == 0.9
        assert kw["max_tokens"] == 128
        assert kw["model"] == "gpt-4-turbo"

    @patch("src.llm.openai_client.OpenAI")
    def test_chat_converts_max_tokens_for_gpt5_models(self, MockOpenAI):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = _make_completion()

        client = OpenAIClient(_make_provider(model="gpt-5.5", max_tokens=320))
        client.chat([{"role": "user", "content": "hi"}])

        kw = mock_client.chat.completions.create.call_args.kwargs
        assert kw["model"] == "gpt-5.5"
        assert kw["max_completion_tokens"] == 320
        assert "temperature" not in kw
        assert "max_tokens" not in kw

    @patch("src.llm.openai_client.OpenAI")
    def test_chat_removes_sampling_params_for_gpt5_models(self, MockOpenAI):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = _make_completion()

        client = OpenAIClient(_make_provider(model="openai/gpt-5.5", max_tokens=320))
        client.chat(
            [{"role": "user", "content": "hi"}],
            temperature=0.1,
            top_p=0.8,
        )

        kw = mock_client.chat.completions.create.call_args.kwargs
        assert kw["model"] == "openai/gpt-5.5"
        assert "temperature" not in kw
        assert "top_p" not in kw

    @patch("src.llm.openai_client.OpenAI")
    def test_chat_simple_returns_string(self, MockOpenAI):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = _make_completion("42")

        client = OpenAIClient(_make_provider())
        result = client.chat_simple("What is 6 * 7?")

        assert result == "42"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# OpenAIClient — retry logic
# ---------------------------------------------------------------------------

class TestOpenAIClientRetry:
    @patch("src.llm.openai_client.time.sleep", return_value=None)
    @patch("src.llm.openai_client.OpenAI")
    def test_retries_on_connection_error(self, MockOpenAI, mock_sleep):
        completion = _make_completion("ok")
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.side_effect = [
            APIConnectionError(request=MagicMock()),
            APIConnectionError(request=MagicMock()),
            completion,
        ]

        client = OpenAIClient(_make_provider(), max_retries=3, retry_delay=0.1)
        resp = client.chat([{"role": "user", "content": "hi"}])

        assert resp.content == "ok"
        assert mock_client.chat.completions.create.call_count == 3

    @patch("src.llm.openai_client.time.sleep", return_value=None)
    @patch("src.llm.openai_client.OpenAI")
    def test_retries_on_timeout_error(self, MockOpenAI, mock_sleep):
        completion = _make_completion("done")
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.side_effect = [
            APITimeoutError(request=MagicMock()),
            completion,
        ]

        client = OpenAIClient(_make_provider(), max_retries=3)
        resp = client.chat([{"role": "user", "content": "hi"}])

        assert resp.content == "done"

    @patch("src.llm.openai_client.time.sleep", return_value=None)
    @patch("src.llm.openai_client.OpenAI")
    def test_raises_after_max_retries(self, MockOpenAI, mock_sleep):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock()
        )

        client = OpenAIClient(_make_provider(), max_retries=2, retry_delay=0.01)

        with pytest.raises(RuntimeError, match="All .* attempts failed"):
            client.chat([{"role": "user", "content": "hi"}])

        assert mock_client.chat.completions.create.call_count == 3  # 1 initial + 2 retries

    @patch("src.llm.openai_client.time.sleep", return_value=None)
    @patch("src.llm.openai_client.OpenAI")
    def test_retry_on_429_rate_limit(self, MockOpenAI, mock_sleep):
        completion = _make_completion("throttled but ok")
        mock_client = MockOpenAI.return_value

        fake_response = MagicMock()
        fake_response.headers = {"retry-after": "0"}
        fake_response.request = MagicMock()

        rate_limit_error = APIStatusError(
            "rate limited",
            response=fake_response,
            body=None,
        )
        rate_limit_error.status_code = 429

        mock_client.chat.completions.create.side_effect = [rate_limit_error, completion]

        client = OpenAIClient(_make_provider(), max_retries=2)
        resp = client.chat([{"role": "user", "content": "hi"}])

        assert resp.content == "throttled but ok"

    @patch("src.llm.openai_client.OpenAI")
    def test_non_retryable_status_error_re_raises(self, MockOpenAI):
        mock_client = MockOpenAI.return_value

        fake_response = MagicMock()
        fake_response.headers = {}
        fake_response.request = MagicMock()

        auth_error = APIStatusError("unauthorized", response=fake_response, body=None)
        auth_error.status_code = 401

        mock_client.chat.completions.create.side_effect = auth_error

        client = OpenAIClient(_make_provider(), max_retries=3)
        with pytest.raises(APIStatusError):
            client.chat([{"role": "user", "content": "hi"}])

        # Should NOT retry on 401
        assert mock_client.chat.completions.create.call_count == 1

    @patch("src.llm.openai_client.OpenAI")
    def test_retries_after_converting_max_tokens_on_400(self, MockOpenAI):
        completion = _make_completion("converted and ok")
        mock_client = MockOpenAI.return_value

        fake_response = MagicMock()
        fake_response.headers = {}
        fake_response.request = MagicMock()

        bad_request = APIStatusError(
            "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
            response=fake_response,
            body=None,
        )
        bad_request.status_code = 400

        mock_client.chat.completions.create.side_effect = [bad_request, completion]

        client = OpenAIClient(_make_provider(model="gpt-4o", max_tokens=512), max_retries=2)
        resp = client.chat([{"role": "user", "content": "hi"}])

        assert resp.content == "converted and ok"
        assert mock_client.chat.completions.create.call_count == 2

        first_call_kw = mock_client.chat.completions.create.call_args_list[0].kwargs
        second_call_kw = mock_client.chat.completions.create.call_args_list[1].kwargs
        assert first_call_kw["max_tokens"] == 512
        assert "max_completion_tokens" not in first_call_kw
        assert second_call_kw["max_completion_tokens"] == 512
        assert "max_tokens" not in second_call_kw

    @patch("src.llm.openai_client.OpenAI")
    def test_retries_after_removing_unsupported_temperature_value(self, MockOpenAI):
        completion = _make_completion("temperature removed")
        mock_client = MockOpenAI.return_value

        fake_response = MagicMock()
        fake_response.headers = {}
        fake_response.request = MagicMock()

        bad_request = APIStatusError(
            "Unsupported value: 'temperature' does not support 0.1 with this model. Only the default (1) value is supported.",
            response=fake_response,
            body={"error": {"param": "temperature"}},
        )
        bad_request.status_code = 400

        mock_client.chat.completions.create.side_effect = [bad_request, completion]

        client = OpenAIClient(_make_provider(model="gpt-4o", temperature=0.1), max_retries=2)
        resp = client.chat([{"role": "user", "content": "hi"}])

        assert resp.content == "temperature removed"
        assert mock_client.chat.completions.create.call_count == 2

        first_call_kw = mock_client.chat.completions.create.call_args_list[0].kwargs
        second_call_kw = mock_client.chat.completions.create.call_args_list[1].kwargs
        assert first_call_kw["temperature"] == 0.1
        assert "temperature" not in second_call_kw

    @patch("src.llm.openai_client.time.sleep", return_value=None)
    @patch("src.llm.openai_client.OpenAI")
    def test_exponential_backoff(self, MockOpenAI, mock_sleep):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.side_effect = [
            APIConnectionError(request=MagicMock()),
            APIConnectionError(request=MagicMock()),
            _make_completion("ok"),
        ]

        client = OpenAIClient(_make_provider(), max_retries=3, retry_delay=1.0)
        client.chat([{"role": "user", "content": "test"}])

        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        # First sleep: 1.0s, second sleep: 2.0s (doubled)
        assert sleep_calls[0] == 1.0
        assert sleep_calls[1] == 2.0


# ---------------------------------------------------------------------------
# OpenAIClient — streaming
# ---------------------------------------------------------------------------

class TestOpenAIClientStream:
    @patch("src.llm.openai_client.OpenAI")
    def test_stream_yields_chunks(self, MockOpenAI):
        mock_client = MockOpenAI.return_value

        def _fake_chunks():
            for text in ["Hello", ", ", "world", "!"]:
                chunk = MagicMock()
                chunk.choices = [SimpleNamespace(delta=SimpleNamespace(content=text))]
                yield chunk

        # stream() uses a context manager
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=_fake_chunks())
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_client.chat.completions.create.return_value = mock_stream_ctx

        client = OpenAIClient(_make_provider())
        result = list(client.stream([{"role": "user", "content": "Say hello"}]))

        assert result == ["Hello", ", ", "world", "!"]

    @patch("src.llm.openai_client.OpenAI")
    def test_stream_skips_none_deltas(self, MockOpenAI):
        mock_client = MockOpenAI.return_value

        def _fake_chunks():
            for text in ["Hi", None, "!"]:
                chunk = MagicMock()
                chunk.choices = [SimpleNamespace(delta=SimpleNamespace(content=text))]
                yield chunk

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=_fake_chunks())
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_client.chat.completions.create.return_value = mock_stream_ctx

        client = OpenAIClient(_make_provider())
        result = list(client.stream([{"role": "user", "content": "hi"}]))

        assert result == ["Hi", "!"]


# ---------------------------------------------------------------------------
# OpenAIClient — repr
# ---------------------------------------------------------------------------

class TestOpenAIClientRepr:
    @patch("src.llm.openai_client.OpenAI")
    def test_repr_contains_provider_and_retries(self, MockOpenAI):
        client = OpenAIClient(_make_provider(), max_retries=5)
        r = repr(client)
        assert "OpenAIClient" in r
        assert "max_retries=5" in r
