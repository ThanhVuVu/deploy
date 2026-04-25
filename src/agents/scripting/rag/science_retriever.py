"""
Scripting-only RAG retriever for grounded science facts.

This module intentionally lives under the scripting agent package so that
RAG logic is isolated from other agents.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import chromadb
except ImportError:  # pragma: no cover - environment-specific dependency
    chromadb = None
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # pragma: no cover - environment-specific dependency
    OpenAIEmbeddings = None
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - environment-specific dependency
    RecursiveCharacterTextSplitter = None
try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - environment-specific dependency
    PdfReader = None

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "agent",
    "and",
    "cho",
    "create",
    "cua",
    "cung",
    "experiment",
    "for",
    "guide",
    "hoc",
    "how",
    "lab",
    "lop",
    "middle",
    "nghiem",
    "phong",
    "script",
    "simulator",
    "sinh",
    "student",
    "teacher",
    "the",
    "thi",
    "this",
    "virtual",
    "with",
    "write",
}

_TOPIC_KEYWORDS: dict[str, set[str]] = {
    "static_electricity": {
        "balloon",
        "charge",
        "charged",
        "dien",
        "dientich",
        "electrostatic",
        "electricity",
        "friction",
        "paper",
        "static",
        "tich",
        "tinh",
        "tĩnh",
        "điện",
    },
    "magnets": {
        "attract",
        "cham",
        "compass",
        "magnet",
        "magnetic",
        "nam",
        "namcham",
        "pole",
        "repel",
        "truong",
        "tu",
        "từ",
        "châm",
        "trường",
    },
    "boiling_water": {
        "boil",
        "boiling",
        "bubble",
        "dun",
        "heat",
        "kpa",
        "nhiet",
        "nuoc",
        "pressure",
        "soi",
        "steam",
        "temperature",
        "water",
        "đun",
        "sôi",
        "nhiệt",
        "nước",
    },
}


@dataclass(frozen=True)
class RetrievedChunk:
    """Single chunk retrieved from the science vector store."""

    chunk_id: str
    source_file: str
    topic: str
    section: str
    distance: float
    content: str


@dataclass(frozen=True)
class RetrievalResult:
    """Container for retriever output used by ScriptingAgent."""

    chunks: list[RetrievedChunk]

    @property
    def has_enough_data(self) -> bool:
        return bool(self.chunks)

    @property
    def grounded_context(self) -> str:
        if not self.chunks:
            return ""

        blocks: list[str] = []
        for idx, chunk in enumerate(self.chunks, start=1):
            blocks.append(
                "\n".join(
                    [
                        (
                            f"[Chunk {idx}] topic={chunk.topic}; source={chunk.source_file}; "
                            f"section={chunk.section}; chunk_id={chunk.chunk_id}; "
                            f"distance={chunk.distance:.4f}"
                        ),
                        chunk.content,
                    ]
                )
            )
        return "\n\n".join(blocks)

    @property
    def source_report(self) -> str:
        if not self.chunks:
            return ""

        return "\n".join(
            (
                f"- {chunk.source_file} | {chunk.section} | "
                f"chunk_id={chunk.chunk_id} | distance={chunk.distance:.4f}"
            )
            for chunk in self.chunks
        )


class ScriptingScienceRetriever:
    """
    Build and query a local Chroma index for scripting-agent science facts.

    The index is fed from ``science_db`` documents.
    Markdown files are preferred; PDF ingestion is used as a fallback.
    """

    def __init__(
        self,
        *,
        database_dir: str | Path = "science_db",
        persist_dir: str | Path = ".rag/scripting_chroma",
        collection_name: str = "scripting_science_facts",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        default_top_k: int = 4,
        max_distance: float = 1.1,
        min_token_overlap: int = 1,
        embedding_model: str = "text-embedding-3-small",
        embedding_client: Optional[Any] = None,
    ) -> None:
        if chromadb is None:
            raise ImportError(
                "chromadb is required for ScriptingScienceRetriever. "
                "Install it with: pip install chromadb"
            )
        if embedding_client is None and OpenAIEmbeddings is None:
            raise ImportError(
                "langchain-openai is required for ScriptingScienceRetriever. "
                "Install it with: pip install langchain-openai"
            )
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain-text-splitters is required for ScriptingScienceRetriever. "
                "Install it with: pip install langchain-text-splitters"
            )

        self.database_dir = Path(database_dir)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.default_top_k = default_top_k
        self.max_distance = max_distance
        self.min_token_overlap = min_token_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n", ". ", " ", ""],
        )
        self.embedding_client = embedding_client or OpenAIEmbeddings(model=embedding_model)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.build_or_update_index()

    def build_or_update_index(self) -> int:
        """Load science facts, chunk them, and upsert missing chunks."""
        chunk_records = self._chunk_records_from_sources()
        if not chunk_records:
            logger.warning(
                "ScriptingScienceRetriever: no supported science documents found at %s",
                self.database_dir,
            )
            return 0

        existing_ids = self._load_existing_ids()
        new_records = [record for record in chunk_records if record["id"] not in existing_ids]

        if not new_records:
            logger.info("ScriptingScienceRetriever: index already up to date.")
            return 0

        texts = [record["content"] for record in new_records]
        vectors = self.embedding_client.embed_documents(texts)

        self.collection.add(
            ids=[record["id"] for record in new_records],
            documents=texts,
            metadatas=[record["metadata"] for record in new_records],
            embeddings=vectors,
        )

        logger.info("ScriptingScienceRetriever: indexed %s new chunks.", len(new_records))
        return len(new_records)

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> RetrievalResult:
        """Return relevant science chunks for a teacher query."""
        if not query.strip():
            return RetrievalResult(chunks=[])

        if self.collection.count() == 0:
            logger.warning("ScriptingScienceRetriever: empty collection.")
            return RetrievalResult(chunks=[])

        query_vector = self.embedding_client.embed_query(query)
        response = self.collection.query(
            query_embeddings=[query_vector],
            n_results=max(1, top_k or self.default_top_k),
            include=["documents", "metadatas", "distances"],
        )

        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if not document or not metadata:
                continue

            score = float(distance if distance is not None else 999.0)
            topic = str(metadata.get("topic", "")).strip().lower()

            if not self._is_relevant(query, document, topic=topic, distance=score):
                continue

            chunks.append(
                RetrievedChunk(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    source_file=str(metadata.get("source_file", "")),
                    topic=topic,
                    section=str(metadata.get("section", "")),
                    distance=score,
                    content=document.strip(),
                )
            )

        return RetrievalResult(chunks=chunks)

    def _load_existing_ids(self) -> set[str]:
        """Get current IDs from Chroma collection with pagination."""
        existing_ids: set[str] = set()
        page_size = 200
        offset = 0

        while True:
            batch = self.collection.get(limit=page_size, offset=offset, include=["metadatas"])
            ids = batch.get("ids", [])
            if not ids:
                break

            existing_ids.update(ids)
            if len(ids) < page_size:
                break

            offset += page_size

        return existing_ids

    def _chunk_records_from_sources(self) -> list[dict[str, Any]]:
        """Collect chunk records from preferred markdown files, then PDF fallback."""
        markdown_records = self._chunk_records_from_markdown()
        if markdown_records:
            return markdown_records

        return self._chunk_records_from_pdf()

    def _chunk_records_from_markdown(self) -> list[dict[str, Any]]:
        """Load markdown files and convert them into chunk records."""
        if not self.database_dir.exists():
            return []

        records: list[dict[str, Any]] = []
        markdown_files = sorted(self.database_dir.glob("*.md")) + sorted(
            self.database_dir.glob("*.markdown")
        )
        for file_path in markdown_files:
            text = file_path.read_text(encoding="utf-8").strip()
            if not text:
                continue

            topic = file_path.stem.lower()
            for section, section_text in self._iter_sections(text):
                records.extend(
                    self._records_from_text(
                        source_file=file_path.name,
                        topic=topic,
                        section=section,
                        text=section_text,
                    )
                )

        return records

    def _chunk_records_from_pdf(self) -> list[dict[str, Any]]:
        """Extract text from PDF files and convert extractable pages into chunks."""
        if not self.database_dir.exists():
            return []

        pdf_files = sorted(self.database_dir.glob("*.pdf"))
        if not pdf_files:
            return []

        if PdfReader is None:
            logger.warning(
                "ScriptingScienceRetriever: pypdf is not installed; skipping %s PDF file(s).",
                len(pdf_files),
            )
            return []

        records: list[dict[str, Any]] = []
        for file_path in pdf_files:
            try:
                reader = PdfReader(str(file_path))
            except Exception as exc:
                logger.warning(
                    "ScriptingScienceRetriever: failed to parse PDF %s: %s",
                    file_path.name,
                    exc,
                )
                continue

            topic = file_path.stem.lower()
            for page_index, page in enumerate(reader.pages, start=1):
                try:
                    page_text = (page.extract_text() or "").strip()
                except Exception as exc:
                    logger.warning(
                        "ScriptingScienceRetriever: failed to extract text from %s page %s: %s",
                        file_path.name,
                        page_index,
                        exc,
                    )
                    continue

                if not page_text:
                    continue

                normalized_text = "\n".join(
                    line.strip() for line in page_text.splitlines() if line.strip()
                )
                if not normalized_text:
                    continue

                records.extend(
                    self._records_from_text(
                        source_file=file_path.name,
                        topic=topic,
                        section=f"page-{page_index}",
                        text=normalized_text,
                    )
                )

        if not records:
            logger.warning(
                "ScriptingScienceRetriever: PDF files found but no extractable text in %s",
                self.database_dir,
            )

        return records

    def _records_from_text(
        self,
        *,
        source_file: str,
        topic: str,
        section: str,
        text: str,
    ) -> list[dict[str, Any]]:
        """Split text into chunks and return Chroma-ready records."""
        records: list[dict[str, Any]] = []
        chunks = self.text_splitter.split_text(text)
        for index, chunk in enumerate(chunks):
            content = chunk.strip()
            if not content:
                continue

            chunk_id = self._build_chunk_id(
                source_file=source_file,
                section=section,
                index=index,
                content=content,
            )

            records.append(
                {
                    "id": chunk_id,
                    "content": content,
                    "metadata": {
                        "topic": topic,
                        "source_file": source_file,
                        "section": section,
                        "chunk_id": chunk_id,
                    },
                }
            )

        return records

    @staticmethod
    def _iter_sections(text: str) -> list[tuple[str, str]]:
        """Split markdown content by H2 headers while keeping deterministic sections."""
        sections: list[tuple[str, str]] = []
        current_title = "overview"
        buffer: list[str] = []

        for line in text.splitlines():
            if line.startswith("## "):
                if buffer:
                    section_text = "\n".join(buffer).strip()
                    if section_text:
                        sections.append((current_title, section_text))
                current_title = line[3:].strip().lower()
                buffer = [line]
            else:
                buffer.append(line)

        if buffer:
            section_text = "\n".join(buffer).strip()
            if section_text:
                sections.append((current_title, section_text))

        return sections

    @staticmethod
    def _build_chunk_id(*, source_file: str, section: str, index: int, content: str) -> str:
        raw = f"{source_file}|{section}|{index}|{content}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _is_relevant(self, query: str, content: str, *, topic: str, distance: float) -> bool:
        if distance > self.max_distance:
            return False

        query_tokens = self._tokenize(query)
        content_tokens = self._tokenize(content)
        if not query_tokens or not content_tokens:
            return False

        if len(query_tokens.intersection(content_tokens)) < self.min_token_overlap:
            return False

        topic_keywords = _TOPIC_KEYWORDS.get(topic, set())
        if topic_keywords and query_tokens.isdisjoint(topic_keywords):
            return False

        return True

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = {
            token
            for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
            if len(token) >= 3
        }
        return {token for token in tokens if token not in _STOPWORDS}
