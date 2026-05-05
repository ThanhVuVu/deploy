"""
src/agents/scripting/rag/multimodal_retriever.py

Unified retrieval across three ChromaDB collections — all in the same
text-embedding-3-small vector space (1536-dim):

  • mm_text_chunks  : raw text chunks from PDF
  • mm_image_chunks : GPT-4o mini vision captions of images
  • mm_table_chunks : GPT-4o mini summaries of tables

A single TextEmbedder query is used for all three collections, so
text, image, and table results are fully comparable and can be
merged and re-ranked in one unified score space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────── Result dataclasses ──────────────────────────────


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    source_file: str
    page_number: int
    section: str
    content: str
    score: float          # higher = more relevant (normalised)
    modality: str = "text"


@dataclass(frozen=True)
class ImageChunk:
    chunk_id: str
    source_file: str
    page_number: int
    image_index: int
    caption: str          # GPT-4o mini generated caption
    score: float
    modality: str = "image"


@dataclass(frozen=True)
class TableChunk:
    chunk_id: str
    source_file: str
    page_number: int
    table_index: int
    summary: str
    original_markdown: str
    score: float
    modality: str = "table"


@dataclass
class MultimodalRetrievalResult:
    """Structured output consumed by ScriptingAgent."""

    text_chunks: list[TextChunk] = field(default_factory=list)
    image_chunks: list[ImageChunk] = field(default_factory=list)
    table_chunks: list[TableChunk] = field(default_factory=list)

    @property
    def has_enough_data(self) -> bool:
        return bool(self.text_chunks or self.table_chunks)

    @property
    def grounded_context(self) -> str:
        """Formatted context string for GPT-4o mini."""
        blocks: list[str] = []
        idx = 1

        for chunk in self.text_chunks:
            blocks.append(
                f"[Text {idx}] source={chunk.source_file} | "
                f"page={chunk.page_number} | section={chunk.section[:80]} | "
                f"score={chunk.score:.3f}\n{chunk.content}"
            )
            idx += 1

        for chunk in self.table_chunks:
            blocks.append(
                f"[Table {idx}] source={chunk.source_file} | "
                f"page={chunk.page_number} | score={chunk.score:.3f}\n"
                f"{chunk.summary}\n\nOriginal Markdown:\n{chunk.original_markdown}"
            )
            idx += 1

        for chunk in self.image_chunks:
            blocks.append(
                f"[Image {idx}] source={chunk.source_file} | "
                f"page={chunk.page_number} | image_index={chunk.image_index} | "
                f"score={chunk.score:.3f}\n"
                f"Mô tả hình ảnh: {chunk.caption or '(không có mô tả)'}"
            )
            idx += 1

        return "\n\n---\n\n".join(blocks)

    @property
    def source_report(self) -> str:
        lines: list[str] = []
        for c in self.text_chunks:
            lines.append(
                f"- [TEXT] {c.source_file} | page={c.page_number} | "
                f"chunk_id={c.chunk_id} | score={c.score:.3f}"
            )
        for c in self.table_chunks:
            lines.append(
                f"- [TABLE] {c.source_file} | page={c.page_number} | "
                f"chunk_id={c.chunk_id} | score={c.score:.3f}"
            )
        for c in self.image_chunks:
            lines.append(
                f"- [IMAGE] {c.source_file} | page={c.page_number} | "
                f"img_idx={c.image_index} | score={c.score:.3f}"
            )
        return "\n".join(lines)


# ──────────────────────────── Retriever ───────────────────────────────────────


class MultimodalRetriever:
    """
    Unified retriever over text, image (caption), and table ChromaDB collections.

    All three collections share the same text-embedding-3-small vector space,
    so a single query vector is sufficient for all modalities.

    Parameters
    ----------
    persist_dir  : Same root dir as MultimodalIndexer (for auto-init).
    text_top_k   : Candidates to pull from text collection.
    image_top_k  : Candidates to pull from image collection.
    table_top_k  : Candidates to pull from table collection.
    max_distance : Cosine distance ceiling; chunks above this are dropped.
    """

    def __init__(
        self,
        *,
        persist_dir: str = ".rag/multimodal_chroma",
        text_top_k: int = 5,
        image_top_k: int = 3,
        table_top_k: int = 3,
        max_distance: float = 1.0,
        # Allow injecting pre-built collections for testing
        col_text=None,
        col_image=None,
        col_table=None,
        text_embedder=None,
    ) -> None:
        self.text_top_k = text_top_k
        self.image_top_k = image_top_k
        self.table_top_k = table_top_k
        self.max_distance = max_distance

        # ── Collections ────────────────────────────────────────────────────
        if col_text is not None:
            self._col_text = col_text
            self._col_image = col_image
            self._col_table = col_table
        else:
            # Auto-init from a shared indexer
            from .multimodal_indexer import MultimodalIndexer

            indexer = MultimodalIndexer(persist_dir=persist_dir)
            self._col_text, self._col_image, self._col_table = (
                indexer.get_collections()
            )

        # Single text embedder for ALL modalities (lazy)
        self._text_embedder = text_embedder

    # ── Public API ─────────────────────────────────────────────────────────

    def hybrid_retrieve(
        self,
        user_query: str,
        top_k: int = 4,
    ) -> MultimodalRetrievalResult:
        """
        Retrieve relevant chunks across all three modalities.

        A single text embedding is used to query text, image-caption,
        and table-summary collections. Results are merged and re-ranked
        by cosine similarity score before returning the top-K.
        """
        if not user_query.strip():
            return MultimodalRetrievalResult()

        # One embedding vector queries ALL three collections
        query_vec = self._get_text_embedder().embed_query(user_query)

        raw_texts = self._query_collection(self._col_text, query_vec, self.text_top_k)
        raw_tables = self._query_collection(self._col_table, query_vec, self.table_top_k)
        raw_images = self._query_collection(self._col_image, query_vec, self.image_top_k)

        # Build typed chunk objects (equal weight — same vector space)
        text_chunks = self._build_text_chunks(raw_texts)
        table_chunks = self._build_table_chunks(raw_tables)
        image_chunks = self._build_image_chunks(raw_images)

        # Unified re-rank across all modalities
        all_items: list[Any] = text_chunks + table_chunks + image_chunks
        all_items.sort(key=lambda x: x.score, reverse=True)

        result_texts: list[TextChunk] = []
        result_tables: list[TableChunk] = []
        result_images: list[ImageChunk] = []

        for item in all_items[:top_k]:
            if isinstance(item, TextChunk):
                result_texts.append(item)
            elif isinstance(item, TableChunk):
                result_tables.append(item)
            elif isinstance(item, ImageChunk):
                result_images.append(item)

        return MultimodalRetrievalResult(
            text_chunks=result_texts,
            image_chunks=result_images,
            table_chunks=result_tables,
        )

    # ── Collection query helper ────────────────────────────────────────────

    def _query_collection(
        self, collection, query_vector: list[float], n_results: int
    ) -> list[dict]:
        """Query a Chroma collection and return raw result rows."""
        if collection is None or collection.count() == 0:
            return []

        n = min(n_results, collection.count())
        try:
            response = collection.query(
                query_embeddings=[query_vector],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("MultimodalRetriever: collection query failed — %s", exc)
            return []

        docs = (response.get("documents") or [[]])[0]
        metas = (response.get("metadatas") or [[]])[0]
        dists = (response.get("distances") or [[]])[0]

        rows = []
        for doc, meta, dist in zip(docs, metas, dists):
            if doc is None or meta is None:
                continue
            dist_val = float(dist) if dist is not None else 1.0
            if dist_val > self.max_distance:
                continue
            rows.append({"doc": doc, "meta": meta, "distance": dist_val})
        return rows

    # ── Chunk builders ────────────────────────────────────────────────────

    def _build_text_chunks(self, rows: list[dict]) -> list[TextChunk]:
        chunks = []
        for row in rows:
            meta = row["meta"]
            score = self._dist_to_score(row["distance"])
            chunks.append(
                TextChunk(
                    chunk_id=str(meta.get("chunk_id", "")),
                    source_file=str(meta.get("source_file", "")),
                    page_number=int(meta.get("page_number", 0)),
                    section=str(meta.get("section", "")),
                    content=str(row["doc"]),
                    score=score,
                )
            )
        return chunks

    def _build_image_chunks(self, rows: list[dict]) -> list[ImageChunk]:
        chunks = []
        for row in rows:
            meta = row["meta"]
            score = self._dist_to_score(row["distance"])
            # caption = the GPT-4o mini generated description (stored as document)
            caption = str(row["doc"])
            chunks.append(
                ImageChunk(
                    chunk_id=str(meta.get("chunk_id", "")),
                    source_file=str(meta.get("source_file", "")),
                    page_number=int(meta.get("page_number", 0)),
                    image_index=int(meta.get("image_index", 0)),
                    caption=caption,
                    score=score,
                )
            )
        return chunks

    def _build_table_chunks(self, rows: list[dict]) -> list[TableChunk]:
        chunks = []
        for row in rows:
            meta = row["meta"]
            score = self._dist_to_score(row["distance"])
            chunks.append(
                TableChunk(
                    chunk_id=str(meta.get("chunk_id", "")),
                    source_file=str(meta.get("source_file", "")),
                    page_number=int(meta.get("page_number", 0)),
                    table_index=int(meta.get("table_index", 0)),
                    summary=str(row["doc"]),
                    original_markdown=str(meta.get("original_markdown", "")),
                    score=score,
                )
            )
        return chunks

    # ── Score normalisation ───────────────────────────────────────────────

    @staticmethod
    def _dist_to_score(distance: float) -> float:
        """Convert cosine distance [0, 2] to similarity score [0, 1]."""
        return max(0.0, 1.0 - distance / 2.0)

    # ── Lazy embedder loaders ─────────────────────────────────────────────

    def _get_text_embedder(self):
        if self._text_embedder is None:
            from .embeddings import TextEmbedder
            self._text_embedder = TextEmbedder()
        return self._text_embedder
