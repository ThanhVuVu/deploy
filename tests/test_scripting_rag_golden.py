"""
Golden retrieval tests for scripting-agent RAG.

These tests validate that queries against the mock science database return
expected grounded facts (golden snippets) and reject out-of-scope topics.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

import src.agents.scripting.rag.science_retriever as retriever_module
from src.agents.scripting.rag.science_retriever import ScriptingScienceRetriever


class _SimpleSplitter:
    """Lightweight fallback splitter for tests."""

    def __init__(self, *, chunk_size: int, chunk_overlap: int, separators: list[str]) -> None:
        self.chunk_size = max(100, chunk_size)

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        return [text[idx : idx + self.chunk_size] for idx in range(0, len(text), self.chunk_size)]


class _KeywordEmbeddings:
    """Deterministic embedding stub for unit tests (no network calls)."""

    _VOCAB = [
        "static",
        "electricity",
        "balloon",
        "charge",
        "coulomb",
        "magnet",
        "pole",
        "compass",
        "attract",
        "repel",
        "boiling",
        "water",
        "pressure",
        "kpa",
        "temperature",
        "steam",
        "heat",
    ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        return [float(tokens.count(term)) for term in self._VOCAB]


class _InMemoryCollection:
    def __init__(self) -> None:
        self._rows: dict[str, dict] = {}

    def add(self, *, ids, documents, metadatas, embeddings) -> None:
        for _id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            self._rows[_id] = {
                "id": _id,
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def count(self) -> int:
        return len(self._rows)

    def get(self, *, limit: int, offset: int, include) -> dict:
        all_ids = list(self._rows.keys())
        return {"ids": all_ids[offset : offset + limit]}

    def query(self, *, query_embeddings, n_results: int, include) -> dict:
        query_vector = query_embeddings[0]
        ranked = sorted(
            self._rows.values(),
            key=lambda row: self._cosine_distance(query_vector, row["embedding"]),
        )
        top = ranked[:n_results]

        return {
            "documents": [[row["document"] for row in top]],
            "metadatas": [[row["metadata"] for row in top]],
            "distances": [[self._cosine_distance(query_vector, row["embedding"]) for row in top]],
        }

    @staticmethod
    def _cosine_distance(vec_a: list[float], vec_b: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 1.0
        similarity = dot / (norm_a * norm_b)
        return 1.0 - similarity


class _InMemoryPersistentClient:
    def __init__(self, path: str) -> None:
        self._collections: dict[str, _InMemoryCollection] = {}

    def get_or_create_collection(self, *, name: str, metadata: dict) -> _InMemoryCollection:
        if name not in self._collections:
            self._collections[name] = _InMemoryCollection()
        return self._collections[name]


class _FakeChromaModule:
    def PersistentClient(self, path: str) -> _InMemoryPersistentClient:
        return _InMemoryPersistentClient(path)


@pytest.fixture()
def golden_retriever(monkeypatch) -> ScriptingScienceRetriever:
    """Build retriever from mock markdown database with fully local fakes."""
    monkeypatch.setattr(retriever_module, "chromadb", _FakeChromaModule())
    if retriever_module.RecursiveCharacterTextSplitter is None:
        monkeypatch.setattr(retriever_module, "RecursiveCharacterTextSplitter", _SimpleSplitter)

    project_root = Path(__file__).resolve().parents[1]
    database_dir = project_root / "mock_science_db"

    return ScriptingScienceRetriever(
        database_dir=database_dir,
        persist_dir=project_root,
        embedding_client=_KeywordEmbeddings(),
        default_top_k=4,
    )


@pytest.mark.parametrize(
    "query,expected_source,expected_terms,min_hits",
    [
        (
            "Create a static electricity lesson using balloon friction and Coulomb force.",
            "static_electricity.md",
            ["static electricity", "quantitative relation", "coulomb law"],
            2,
        ),
        (
            "Design a magnet activity showing pole attraction and compass deflection.",
            "magnets.md",
            ["compass", "pole interaction", "magnets move together"],
            2,
        ),
        (
            "Write a boiling water experiment at 101.3 kPa with temperature cues.",
            "boiling_water.md",
            ["101.3", "100.0 c", "boiling occurs"],
            2,
        ),
    ],
)
def test_golden_retrieval_from_mock_database(
    golden_retriever: ScriptingScienceRetriever,
    query: str,
    expected_source: str,
    expected_terms: list[str],
    min_hits: int,
):
    result = golden_retriever.retrieve(query, top_k=4)

    assert result.has_enough_data is True
    assert any(chunk.source_file == expected_source for chunk in result.chunks)

    grounded = result.grounded_context.lower()
    hits = sum(1 for term in expected_terms if term in grounded)
    assert hits >= min_hits


def test_out_of_scope_query_returns_no_data(golden_retriever: ScriptingScienceRetriever):
    result = golden_retriever.retrieve(
        "Create a photosynthesis experiment about chlorophyll and stomata.",
        top_k=4,
    )

    assert result.has_enough_data is False
    assert result.grounded_context == ""


def test_index_rebuild_is_idempotent(golden_retriever: ScriptingScienceRetriever):
    added_chunks = golden_retriever.build_or_update_index()
    assert added_chunks == 0
