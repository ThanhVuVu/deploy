"""
tests/test_scripting_agent_rag.py

Unit tests for scripting-agent RAG integration and retriever helper logic.
These tests avoid network/API calls by using stubs and direct helper checks.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agents.scripting.rag.science_retriever import (
    RetrievalResult,
    RetrievedChunk,
    ScriptingScienceRetriever,
)
from src.agents.scripting.scripting_agent import ScriptingAgent


class _FakeRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self._result = result
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def retrieve(self, query: str, *, top_k: int | None = None) -> RetrievalResult:
        self.last_query = query
        self.last_top_k = top_k
        return self._result


def _make_agent(*, chain: MagicMock | None, retriever: object | None) -> ScriptingAgent:
    """Create an agent instance without running heavy __init__ dependencies."""
    agent = ScriptingAgent.__new__(ScriptingAgent)
    agent.chain = chain
    agent.retriever = retriever
    return agent


class TestScriptingAgentRagFlow:
    def test_run_returns_chain_missing_error(self):
        agent = _make_agent(chain=None, retriever=MagicMock())

        result = agent.run("Create a magnet experiment")

        assert result == "ScriptingAgent is not properly configured (chain missing)."

    def test_run_returns_no_data_when_retriever_missing(self):
        agent = _make_agent(chain=MagicMock(), retriever=None)

        result = agent.run("Create a static electricity experiment")

        assert result == ScriptingAgent._NO_DATA_MESSAGE

    def test_run_returns_no_data_when_retrieval_is_empty(self):
        retriever = _FakeRetriever(RetrievalResult(chunks=[]))
        chain = MagicMock()
        agent = _make_agent(chain=chain, retriever=retriever)

        result = agent.run("Write a boiling-water lab script", context={"rag_top_k": 3})

        assert result == ScriptingAgent._NO_DATA_MESSAGE
        assert retriever.last_query == "Write a boiling-water lab script"
        assert retriever.last_top_k == 3
        chain.invoke.assert_not_called()

    def test_run_injects_grounded_context_when_retrieval_exists(self):
        chunk = RetrievedChunk(
            chunk_id="chunk-1",
            source_file="magnets.md",
            topic="magnets",
            section="core rules",
            distance=0.12,
            content="Opposite poles attract and same poles repel.",
        )
        retriever = _FakeRetriever(RetrievalResult(chunks=[chunk]))
        chain = MagicMock()
        chain.invoke.return_value = "grounded script"
        agent = _make_agent(chain=chain, retriever=retriever)

        result = agent.run(
            "Create a magnet lesson",
            context={"class_level": "middle school", "rag_top_k": "2"},
        )

        assert result == "grounded script"
        assert retriever.last_query == "Create a magnet lesson"
        assert retriever.last_top_k == 2

        payload = chain.invoke.call_args.args[0]
        assert payload["query"] == "Create a magnet lesson"
        assert "class_level: middle school" in payload["context_notes"]
        assert "Opposite poles attract" in payload["retrieved_context"]
        assert "magnets.md" in payload["retrieved_sources"]


class TestScriptingAgentTopK:
    @pytest.mark.parametrize(
        "context, expected",
        [
            ({}, 4),
            ({"rag_top_k": "1"}, 1),
            ({"rag_top_k": 0}, 1),
            ({"rag_top_k": 20}, 8),
            ({"rag_top_k": "invalid"}, 4),
        ],
    )
    def test_resolve_top_k_bounds(self, context, expected):
        assert ScriptingAgent._resolve_top_k(context) == expected


class TestRetrievalResultFormatting:
    def test_grounded_context_and_source_report_render_expected_fields(self):
        chunk = RetrievedChunk(
            chunk_id="abc123",
            source_file="boiling_water.md",
            topic="boiling_water",
            section="core rules",
            distance=0.31,
            content="At 101.3 kPa, pure water boils at 100.0 C.",
        )
        result = RetrievalResult(chunks=[chunk])

        assert result.has_enough_data is True
        assert "topic=boiling_water" in result.grounded_context
        assert "chunk_id=abc123" in result.grounded_context
        assert "boiling_water.md" in result.source_report

    def test_empty_result_is_marked_as_not_enough_data(self):
        result = RetrievalResult(chunks=[])

        assert result.has_enough_data is False
        assert result.grounded_context == ""
        assert result.source_report == ""


class TestScienceRetrieverHelpers:
    def _make_retriever_for_logic_tests(self) -> ScriptingScienceRetriever:
        retriever = ScriptingScienceRetriever.__new__(ScriptingScienceRetriever)
        retriever.max_distance = 1.1
        retriever.min_token_overlap = 1
        return retriever

    def test_iter_sections_splits_by_h2_headers(self):
        text = """# Magnet Notes
Intro paragraph.
## Core Rules
N-S attract.
## Experiments
Compass deflects near magnets.
"""

        sections = ScriptingScienceRetriever._iter_sections(text)

        assert len(sections) == 3
        assert sections[0][0] == "overview"
        assert sections[1][0] == "core rules"
        assert sections[2][0] == "experiments"

    def test_build_chunk_id_is_deterministic(self):
        id_1 = ScriptingScienceRetriever._build_chunk_id(
            source_file="magnets.md",
            section="core rules",
            index=0,
            content="Opposite poles attract.",
        )
        id_2 = ScriptingScienceRetriever._build_chunk_id(
            source_file="magnets.md",
            section="core rules",
            index=0,
            content="Opposite poles attract.",
        )
        id_3 = ScriptingScienceRetriever._build_chunk_id(
            source_file="magnets.md",
            section="core rules",
            index=1,
            content="Opposite poles attract.",
        )

        assert id_1 == id_2
        assert id_1 != id_3

    def test_is_relevant_accepts_matching_topic_and_overlap(self):
        retriever = self._make_retriever_for_logic_tests()

        is_relevant = retriever._is_relevant(
            query="Design a magnet and compass activity",
            content="A compass needle deflects when a bar magnet moves closer.",
            topic="magnets",
            distance=0.25,
        )

        assert is_relevant is True

    def test_is_relevant_rejects_large_distance(self):
        retriever = self._make_retriever_for_logic_tests()

        is_relevant = retriever._is_relevant(
            query="Design a magnet and compass activity",
            content="A compass needle deflects when a bar magnet moves closer.",
            topic="magnets",
            distance=9.0,
        )

        assert is_relevant is False

    def test_is_relevant_rejects_topic_mismatch(self):
        retriever = self._make_retriever_for_logic_tests()

        is_relevant = retriever._is_relevant(
            query="Design a boiling water experiment with pressure values",
            content="Opposite magnetic poles attract and similar poles repel.",
            topic="magnets",
            distance=0.20,
        )

        assert is_relevant is False

    def test_tokenize_keeps_vietnamese_keywords(self):
        tokens = ScriptingScienceRetriever._tokenize("Đun sôi nước ở 101.3 kPa")

        assert "đun" in tokens
        assert "sôi" in tokens
        assert "nước" in tokens
