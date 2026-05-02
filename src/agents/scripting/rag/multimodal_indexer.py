"""
src/agents/scripting/rag/multimodal_indexer.py

Multimodal ingestion pipeline for the Virtual Lab scripting agent.

Three separate ChromaDB collections — all using text-embedding-3-small (1536-dim):
  mm_text_chunks  — semantic text chunks
  mm_image_chunks — GPT-4o mini vision captions of images
  mm_table_chunks — GPT-4o mini summaries of tables

By sharing the same embedding space, a single TextEmbedder query retrieves
relevant text, images, AND tables simultaneously.

Persistent ingestion
--------------------
Each PDF is fingerprinted with SHA256.  A manifest is stored at
  <persist_dir>/ingest_manifest.json
If a PDF's hash is already recorded → SKIP ENTIRELY (no parse, no embed,
no API call).  Ingest once, use forever.

Cache files
-----------
  table_summary_cache.json   — GPT-4o mini table summaries (keyed by table markdown SHA256)
  image_caption_cache.json   — GPT-4o mini image captions  (keyed by image bytes SHA256)
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────── Stats ───────────────────────────────────────────


@dataclass
class IngestStats:
    """Statistics returned after ingesting one PDF."""

    pdf_name: str
    skipped: bool = False          # True if already in manifest → skip
    texts_added: int = 0
    images_added: int = 0
    tables_added: int = 0

    @property
    def total_added(self) -> int:
        return self.texts_added + self.images_added + self.tables_added

    def __str__(self) -> str:
        if self.skipped:
            return f"[{self.pdf_name}] Already ingested — skipped."
        return (
            f"[{self.pdf_name}] Added → "
            f"text={self.texts_added}, "
            f"image={self.images_added}, "
            f"table={self.tables_added}"
        )


# ──────────────────────────── Indexer ─────────────────────────────────────────


class MultimodalIndexer:
    """
    Ingest PDFs into three separate ChromaDB collections.

    Parameters
    ----------
    persist_dir     : Root directory for ChromaDB data + manifest + cache.
    text_chunk_size : Max chars per text chunk (semantic splitting).
    text_chunk_overlap : Overlap between adjacent text chunks.
    openai_model    : LLM model for table/image summarisation.
    embedding_model : OpenAI text embedding model name.
    """

    # ChromaDB collection names — fixed constants, never change
    COLLECTION_TEXT = "mm_text_chunks"
    COLLECTION_IMAGE = "mm_image_chunks"
    COLLECTION_TABLE = "mm_table_chunks"

    # Files inside persist_dir
    MANIFEST_FILE = "ingest_manifest.json"
    TABLE_CACHE_FILE = "table_summary_cache.json"
    IMAGE_CACHE_FILE = "image_caption_cache.json"

    def __init__(
        self,
        *,
        persist_dir: str | Path = ".rag/multimodal_chroma",
        text_chunk_size: int = 700,
        text_chunk_overlap: int = 120,
        openai_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        # ── Validate + initialise lazy dependencies ────────────────────────
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("chromadb is required: pip install chromadb") from exc

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise ImportError(
                "langchain-text-splitters is required"
            ) from exc

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.openai_model = openai_model

        # Text splitter (shared for text and table summary chunks)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_chunk_size,
            chunk_overlap=text_chunk_overlap,
            separators=["\n## ", "\n### ", "\n", ". ", " ", ""],
        )

        # TextEmbedder — single embedder for ALL modalities (lazy)
        self._text_embedder = None
        self._embedding_model = embedding_model
        self._openai_client = None

        # ── ChromaDB client + 3 collections ──────────────────────────────
        # All three collections use the same embedding space (1536-dim OpenAI)
        chroma_path = self.persist_dir / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=str(chroma_path))

        self._col_text = self._chroma.get_or_create_collection(
            name=self.COLLECTION_TEXT,
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
        )
        self._col_image = self._chroma.get_or_create_collection(
            name=self.COLLECTION_IMAGE,
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
        )
        self._col_table = self._chroma.get_or_create_collection(
            name=self.COLLECTION_TABLE,
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
        )

        # ── Persistent manifest + caches ─────────────────────────────────
        self._manifest: dict[str, str] = self._load_json(
            self.persist_dir / self.MANIFEST_FILE
        )
        self._table_cache: dict[str, str] = self._load_json(
            self.persist_dir / self.TABLE_CACHE_FILE
        )
        self._image_cache: dict[str, str] = self._load_json(
            self.persist_dir / self.IMAGE_CACHE_FILE
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: Path) -> IngestStats:
        """
        Ingest a PDF file.  Skips entirely if already recorded in manifest.

        Returns an IngestStats object with counts of added chunks.
        """
        pdf_path = Path(pdf_path)
        stats = IngestStats(pdf_name=pdf_path.name)

        # ── Persistent-ingest guard (SHA256 fingerprint) ──────────────────
        pdf_hash = self._sha256_file(pdf_path)
        if self._manifest.get(pdf_path.name) == pdf_hash:
            logger.info(
                "MultimodalIndexer: '%s' already ingested (hash match) — skip.",
                pdf_path.name,
            )
            stats.skipped = True
            return stats

        logger.info("MultimodalIndexer: ingesting '%s' …", pdf_path.name)

        # ── Parse PDF ─────────────────────────────────────────────────────
        from .pdf_parser import PDFParser

        parser = PDFParser()
        parsed = parser.parse(pdf_path)

        # ── Index text (non-scan PDFs) ────────────────────────────────────
        stats.texts_added = self._index_texts(parsed.texts, pdf_path.name)

        # ── Index images (GPT-4o Vision bóc tách từng trang scan) ─────────
        t, im, tb = self._index_images(parsed.images, pdf_path.name)
        stats.texts_added += t
        stats.images_added = im
        stats.tables_added += tb

        # ── Index tables (non-scan PDFs) ──────────────────────────────────
        stats.tables_added += self._index_tables(parsed.tables, pdf_path.name)

        # ── Record in manifest ────────────────────────────────────────────
        self._manifest[pdf_path.name] = pdf_hash
        self._save_json(self.persist_dir / self.MANIFEST_FILE, self._manifest)

        logger.info("MultimodalIndexer: %s", stats)
        return stats

    def collection_counts(self) -> dict[str, int]:
        """Return current document counts for all three collections."""
        return {
            self.COLLECTION_TEXT: self._col_text.count(),
            self.COLLECTION_IMAGE: self._col_image.count(),
            self.COLLECTION_TABLE: self._col_table.count(),
        }

    def get_collections(self):
        """Return a tuple (col_text, col_image, col_table) for the retriever."""
        return self._col_text, self._col_image, self._col_table

    # ── Text indexing ─────────────────────────────────────────────────────────

    def _index_texts(self, text_elements, pdf_name: str) -> int:
        """Chunk and embed text elements into mm_text_chunks."""
        from .pdf_parser import TextElement

        records: list[dict] = []
        for elem in text_elements:
            chunks = self._splitter.split_text(elem.text)
            for idx, chunk in enumerate(chunks):
                content = chunk.strip()
                if not content:
                    continue
                chunk_id = self._make_id(f"text|{pdf_name}|{elem.page_number}|{idx}|{content}")
                records.append(
                    {
                        "id": chunk_id,
                        "content": content,
                        "metadata": {
                            "type": "text",
                            "source_file": pdf_name,
                            "page_number": elem.page_number,
                            "section": elem.section_hint[:200] if elem.section_hint else "",
                            "chunk_id": chunk_id,
                        },
                    }
                )

        return self._upsert_text_records(records, self._col_text)

    # ── Image / Scan-page indexing ───────────────────────────────────────────

    def _index_images(self, image_elements, pdf_name: str) -> tuple[int, int, int]:
        """
        Dùng GPT-4o Vision để bóc tách từng trang/hình ảnh thành 3 loại:
          texts  → mm_text_chunks
          tables → mm_table_chunks
          images → mm_image_chunks

        Kết quả được cache theo SHA256 của ảnh.
        Trả về (texts_added, images_added, tables_added).
        """
        text_records:  list[dict] = []
        image_records: list[dict] = []
        table_records: list[dict] = []
        total = len(image_elements)

        for i, elem in enumerate(image_elements, start=1):
            img_hash = self._sha256_bytes(elem.image_bytes)

            # ── Lấy từ cache hoặc gọi GPT-4o Vision ─────────────────────
            cached = self._image_cache.get(img_hash)
            if cached is None:
                logger.info(
                    "MultimodalIndexer: GPT-4o bóc tách trang %d/%d (%s)...",
                    i, total, pdf_name,
                )
                layout = self._analyze_page_with_vision(
                    image_bytes=elem.image_bytes,
                    page_number=elem.page_number,
                    pdf_name=pdf_name,
                )
                if layout:
                    cached = json.dumps(layout, ensure_ascii=False)
                    self._image_cache[img_hash] = cached
                    self._save_json(
                        self.persist_dir / self.IMAGE_CACHE_FILE, self._image_cache
                    )

            if not cached:
                continue

            try:
                data = json.loads(cached)
            except Exception:
                continue

            page = elem.page_number

            # ── 1. Văn bản → col_text ─────────────────────────────────────
            for t_idx, block in enumerate(data.get("texts", [])):
                content = block.get("content", "").strip()
                if len(content) < 20:
                    continue
                for c_idx, chunk in enumerate(self._splitter.split_text(content)):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    cid = self._make_id(f"scan_text|{pdf_name}|{page}|{t_idx}|{c_idx}")
                    text_records.append({
                        "id": cid, "content": chunk,
                        "metadata": {
                            "type": "text", "source_file": pdf_name,
                            "page_number": page, "section": "", "chunk_id": cid,
                        },
                    })

            # ── 2. Bảng biểu → col_table ─────────────────────────────────
            for b_idx, tbl in enumerate(data.get("tables", [])):
                md = tbl.get("markdown", "").strip()
                if not md:
                    continue
                cid = self._make_id(f"scan_table|{pdf_name}|{page}|{b_idx}")
                table_records.append({
                    "id": cid, "content": md,
                    "metadata": {
                        "type": "table", "source_file": pdf_name,
                        "page_number": page, "table_index": b_idx,
                        "original_markdown": md[:1000], "chunk_id": cid,
                    },
                })

            # ── 3. Hình vẽ/sơ đồ → col_image ────────────────────────────
            for i_idx, img in enumerate(data.get("images", [])):
                desc = img.get("description", "").strip()
                if not desc:
                    continue
                cid = self._make_id(f"scan_image|{pdf_name}|{page}|{i_idx}")
                image_records.append({
                    "id": cid, "content": desc,
                    "metadata": {
                        "type": "image", "source_file": pdf_name,
                        "page_number": page, "image_index": i_idx, "chunk_id": cid,
                    },
                })

        texts_added  = self._upsert_text_records(text_records,  self._col_text)
        images_added = self._upsert_text_records(image_records, self._col_image)
        tables_added = self._upsert_text_records(table_records, self._col_table)
        return texts_added, images_added, tables_added

    def _analyze_page_with_vision(
        self, *, image_bytes: bytes, page_number: int, pdf_name: str
    ) -> dict | None:
        """
        Gọi GPT-4o Vision với response_format=json_object để bóc tách
        một trang scan thành {texts, tables, images}.
        """
        try:
            from openai import OpenAI

            b64  = base64.b64encode(image_bytes).decode("ascii")
            mime = "image/png" if image_bytes[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"

            prompt = (
                "Bạn là chuyên gia số hóa sách giáo khoa Khoa học Tự nhiên THCS Việt Nam.\n"
                "Phân tích trang sách scan này, trả về JSON đúng cấu trúc sau:\n"
                "{\n"
                '  "texts":  [{"content": "toàn bộ văn bản lý thuyết, định nghĩa, bài tập..."}],\n'
                '  "tables": [{"markdown": "| cột1 | cột2 |\\n|---|---|\\n| gt | gt |"}],\n'
                '  "images": [{"description": "mô tả chi tiết hình vẽ/sơ đồ thí nghiệm..."}]\n'
                "}\n"
                "Quy tắc BẮT BUỘC:\n"
                "1. OCR chính xác tiếng Việt có đầy đủ dấu (ưu tiên số 1).\n"
                "2. Mỗi bảng biểu → chuyển thành Markdown table chuẩn.\n"
                "3. Mỗi hình vẽ/sơ đồ → mô tả 3-5 câu nêu rõ dụng cụ, hiện tượng, cấu tạo.\n"
                "4. Không có bảng → tables: [].  Không có hình → images: [].\n"
                "CHỈ trả về JSON, không thêm bất kỳ text nào khác."
            )

            if self._openai_client is None:
                from openai import OpenAI
                self._openai_client = OpenAI()

            client = self._openai_client
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=self.openai_model,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Trang {page_number} — {pdf_name}:\n{prompt}"},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                    "detail": "high",
                                }},
                            ],
                        }],
                        response_format={"type": "json_object"},
                        max_tokens=3000,
                        temperature=0.0 if attempt == 0 else 0.2, # slight variation on retry
                    )
                    
                    content = response.choices[0].message.content
                    if content is None:
                        finish_reason = response.choices[0].finish_reason
                        refusal = getattr(response.choices[0].message, "refusal", None)
                        if refusal:
                            logger.warning(
                                "MultimodalIndexer: Vision analysis refused (p%s %s). Refusal: %s",
                                page_number, pdf_name, refusal
                            )
                            # Do not skip: return a valid JSON with the refusal message
                            return {
                                "texts": [{"content": f"[CẢNH BÁO: OpenAI từ chối bóc tách hình ảnh này. Lý do: {refusal}]"}],
                                "tables": [],
                                "images": []
                            }
                            
                        logger.warning(
                            "MultimodalIndexer: Vision response content is None (p%s %s, attempt %d). Finish reason: %s",
                            page_number, pdf_name, attempt + 1, finish_reason
                        )
                        import time
                        time.sleep(2)
                        continue # retry
                        
                    return json.loads(content)
                except Exception as exc:
                    logger.warning(
                        "MultimodalIndexer: Vision analysis attempt %d failed (p%s %s) — %s",
                        attempt + 1, page_number, pdf_name, exc
                    )
                    if attempt == max_retries - 1:
                        logger.error("MultimodalIndexer: Vision analysis completely failed after %d retries.", max_retries)
                        raise # Do not skip, bubble up the error
                    import time
                    time.sleep(2)
                    
            raise RuntimeError(f"Vision analysis failed for p{page_number} after {max_retries} retries due to None content.")
        except Exception as exc:
            logger.error(
                "MultimodalIndexer: Fatal error in vision analysis (p%s %s) — %s",
                page_number, pdf_name, exc,
            )
            raise # Strict mode: do not skip any page

    # ── Table indexing ────────────────────────────────────────────────────────

    def _index_tables(self, table_elements, pdf_name: str) -> int:
        """Summarise tables with GPT-4o mini (cached), then embed as text."""
        records: list[dict] = []

        for elem in table_elements:
            md_hash = self._sha256_str(elem.markdown)
            # Try cache first
            summary = self._table_cache.get(md_hash)
            if summary is None:
                summary = self._summarise_table(elem.markdown)
                if summary:
                    self._table_cache[md_hash] = summary
                    self._save_json(
                        self.persist_dir / self.TABLE_CACHE_FILE, self._table_cache
                    )

            if not summary:
                continue

            chunk_id = self._make_id(
                f"table|{pdf_name}|{elem.page_number}|{elem.table_index}"
            )
            records.append(
                {
                    "id": chunk_id,
                    "content": summary,
                    "metadata": {
                        "type": "table",
                        "source_file": pdf_name,
                        "page_number": elem.page_number,
                        "table_index": elem.table_index,
                        "original_markdown": elem.markdown[:1000],
                        "chunk_id": chunk_id,
                    },
                }
            )

        return self._upsert_text_records(records, self._col_table)

    def _summarise_table(self, markdown: str) -> str:
        """Call GPT-4o mini to summarise a Markdown table. Returns empty on failure."""
        try:
            from openai import OpenAI

            if self._openai_client is None:
                from openai import OpenAI
                self._openai_client = OpenAI()

            client = self._openai_client
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a science education assistant. "
                            "Summarise the following Markdown table in 2–4 clear sentences "
                            "that capture the key data, trends, and units. "
                            "Write in Vietnamese if the table content is Vietnamese."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Table:\n{markdown}",
                    },
                ],
                max_tokens=300,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("MultimodalIndexer: table summarisation failed — %s", exc)
            return ""

    # ── Upsert helper (text collections) ─────────────────────────────────────

    def _upsert_text_records(self, records: list[dict], collection) -> int:
        """Embed and upsert records that are not yet in the collection."""
        if not records:
            return 0

        # ── 1. Deduplicate input records by ID ──────────────────────────
        unique_map = {r["id"]: r for r in records}
        unique_records = list(unique_map.values())

        # ── 2. Filter out already existing IDs (optional but saves embedding cost) 
        existing_ids = self._get_existing_ids(collection)
        new_records = [r for r in unique_records if r["id"] not in existing_ids]
        
        if not new_records:
            return 0

        embedder = self._get_text_embedder()
        texts = [r["content"] for r in new_records]

        try:
            vectors = embedder.embed_texts(texts)
        except Exception as exc:
            logger.error("MultimodalIndexer: text embedding failed — %s", exc)
            return 0

        # ── 3. Use upsert instead of add for maximum robustness ─────────
        collection.upsert(
            ids=[r["id"] for r in new_records],
            documents=texts,
            metadatas=[r["metadata"] for r in new_records],
            embeddings=vectors,
        )
        return len(new_records)

    # ── Lazy embedder loaders ─────────────────────────────────────────────────

    def _get_text_embedder(self):
        if self._text_embedder is None:
            from .embeddings import TextEmbedder
            self._text_embedder = TextEmbedder(model=self._embedding_model)
        return self._text_embedder

    # ── ID helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _get_existing_ids(collection) -> set[str]:
        """Paginate through all IDs in a Chroma collection."""
        existing: set[str] = set()
        page_size = 200
        offset = 0
        while True:
            batch = collection.get(limit=page_size, offset=offset, include=[])
            ids = batch.get("ids", [])
            if not ids:
                break
            existing.update(ids)
            if len(ids) < page_size:
                break
            offset += page_size
        return existing

    @staticmethod
    def _make_id(raw: str) -> str:
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _sha256_str(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _sha256_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    # ── JSON persistence helpers ──────────────────────────────────────────────

    @staticmethod
    def _load_json(path: Path) -> dict:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("MultimodalIndexer: could not load %s — %s", path, exc)
        return {}

    @staticmethod
    def _save_json(path: Path, data: dict) -> None:
        try:
            from filelock import FileLock
            lock_path = path.with_suffix(path.suffix + ".lock")
            with FileLock(str(lock_path), timeout=10):
                # Prevent race condition by merging with existing data
                if path.exists():
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                            if isinstance(existing_data, dict):
                                existing_data.update(data)
                                data = existing_data
                    except (json.JSONDecodeError, OSError):
                        pass

                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except ImportError:
            # Fallback if filelock is not installed
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except OSError as exc:
                logger.error("MultimodalIndexer: could not save %s — %s", path, exc)
        except Exception as exc:
            logger.error("MultimodalIndexer: could not save %s — %s", path, exc)
