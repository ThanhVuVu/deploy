"""
tests/test_llm_integration.py

Optional LIVE integration test — makes a real API call to the NVIDIA endpoint.
Requires NVIDIA_API_KEY to be set (reads from .env automatically).

Skip this file during CI or when no key is available:
    pytest tests/ -v --ignore=tests/test_llm_integration.py

Run only the integration test:
    pytest tests/test_llm_integration.py -v -s
"""

from __future__ import annotations

import os
import pytest
from dotenv import load_dotenv

load_dotenv()

# Skip the entire module if no API key is present
pytestmark = pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY not set — skipping live integration tests",
)


from src.llm.provider import get_provider
from src.llm.openai_client import OpenAIClient, LLMResponse
from src.llm.output_parser import OutputParser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> OpenAIClient:
    """A real client pointed at the NVIDIA NIM endpoint."""
    provider = get_provider(
        model="meta/llama-3.1-8b-instruct",  # fast / cheap model for testing
        temperature=0.1,
        max_tokens=256,
    )
    return OpenAIClient(provider, max_retries=2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLiveChat:
    def test_basic_response(self, client):
        resp = client.chat([{"role": "user", "content": "Reply with exactly: OK"}])
        assert isinstance(resp, LLMResponse)
        assert len(resp.content) > 0

    def test_usage_returned(self, client):
        resp = client.chat([{"role": "user", "content": "Say hello."}])
        assert resp.usage.get("total_tokens", 0) > 0

    def test_finish_reason_is_stop(self, client):
        resp = client.chat([{"role": "user", "content": "Say: done"}])
        assert resp.finish_reason in ("stop", "eos_token", None)

    def test_system_prompt(self, client):
        resp = client.chat(
            [{"role": "user", "content": "What are you?"}],
            system_prompt="You are a pirate. Always respond in pirate speak.",
        )
        # Can't assert exact wording, but should be non-empty
        assert len(resp.content) > 0

    def test_chat_simple(self, client):
        result = client.chat_simple("What is 2 + 2? Reply with only the number.")
        assert "4" in result

    def test_stream_assembles_full_response(self, client):
        chunks = list(
            client.stream([{"role": "user", "content": "Count to 3, one word per line."}])
        )
        full = "".join(chunks)
        assert len(full) > 0


class TestLiveOutputParser:
    def test_json_extraction_from_model(self, client):
        prompt = (
            'Respond ONLY with a JSON object, no extra text.\n'
            'Format: {"subject": "...", "fact": "..."}\n'
            'Topic: gravity'
        )
        resp = client.chat_simple(prompt)
        result = OutputParser.parse(resp)
        # Model may or may not wrap it; parser should handle both
        if result.json:
            assert "subject" in result.json or "fact" in result.json

    def test_code_block_extraction_from_model(self, client):
        prompt = (
            "Write a Python function that returns the sum of two numbers. "
            "Use a ```python code block."
        )
        resp = client.chat_simple(prompt)
        result = OutputParser.parse(resp)
        code = result.first_block("python")
        assert code is not None
        assert "def " in code
