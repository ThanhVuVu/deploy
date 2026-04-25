"""Command-line demo for scripting prompt -> RAG retrieval -> grounded output.

This script stays inside src/agents/scripting and does not require importing
src.agents package roots, so it can run even when optional LLM dependencies
are not available.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from pathlib import Path
from typing import Any


class _SimpleSplitter:
    """Fallback splitter when langchain text splitters are unavailable."""

    def __init__(self, *, chunk_size: int, chunk_overlap: int, separators: list[str]) -> None:
        self.chunk_size = max(100, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for idx in range(0, len(text), step):
            chunk = text[idx : idx + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks


class _KeywordEmbeddings:
    """Deterministic local embeddings for offline/demo usage."""

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
    """Minimal Chroma-compatible in-memory collection for local demo."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

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

    def get(self, *, limit: int, offset: int, include) -> dict[str, Any]:
        all_ids = list(self._rows.keys())
        return {"ids": all_ids[offset : offset + limit]}

    def query(self, *, query_embeddings, n_results: int, include) -> dict[str, list[list[Any]]]:
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

    def get_or_create_collection(self, *, name: str, metadata: dict[str, Any]) -> _InMemoryCollection:
        if name not in self._collections:
            self._collections[name] = _InMemoryCollection()
        return self._collections[name]


class _FakeChromaModule:
    def PersistentClient(self, path: str) -> _InMemoryPersistentClient:
        return _InMemoryPersistentClient(path)


def _load_retriever_module():
    """Load science_retriever.py directly from path to avoid package import side effects."""
    module_path = Path(__file__).resolve().parent / "rag" / "science_retriever.py"
    spec = importlib.util.spec_from_file_location("scripting_science_retriever", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load retriever module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_local_backends(retriever_module: Any) -> None:
    """Use deterministic in-memory backends for reproducible local demos."""
    retriever_module.chromadb = _FakeChromaModule()

    if getattr(retriever_module, "RecursiveCharacterTextSplitter", None) is None:
        retriever_module.RecursiveCharacterTextSplitter = _SimpleSplitter


def _extract_brief_facts(text: str, max_sentences: int = 2) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]
    if not sentences:
        return [normalized]
    return sentences[:max_sentences]


_DEMO_STOPWORDS = {
    "a",
    "about",
    "activity",
    "an",
    "and",
    "create",
    "experiment",
    "for",
    "show",
    "showing",
    "the",
    "with",
}


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _clean_markdown_line(line: str) -> str:
    cleaned = line.strip()
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    cleaned = re.sub(r"^[-*]\s*", "", cleaned)
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
    cleaned = _normalize_whitespace(cleaned)
    return cleaned.strip(" -")


def _split_h2_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_title = "overview"
    buffer: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if buffer:
                sections[current_title] = "\n".join(buffer).strip()
            current_title = line[3:].strip().lower()
            buffer = [line]
        else:
            buffer.append(line)

    if buffer:
        sections[current_title] = "\n".join(buffer).strip()

    return sections


def _split_h3_blocks(section_text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current_title = "General"
    buffer: list[str] = []

    for line in section_text.splitlines():
        if line.startswith("### "):
            if buffer:
                body = "\n".join(buffer).strip()
                if body:
                    blocks.append((current_title, body))
            current_title = line[4:].strip()
            buffer = []
        else:
            buffer.append(line)

    if buffer:
        body = "\n".join(buffer).strip()
        if body:
            blocks.append((current_title, body))

    return blocks


def _query_terms(text: str) -> set[str]:
    terms = {
        token
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(token) >= 3
    }
    return {term for term in terms if term not in _DEMO_STOPWORDS}


def _select_best_h3_block(section_text: str, query: str) -> tuple[str, str]:
    blocks = _split_h3_blocks(section_text)
    if not blocks:
        return "General", section_text

    terms = _query_terms(query)

    def score_block(block: tuple[str, str]) -> int:
        title, body = block
        pool = f"{title} {body}".lower()
        token_pool = set(re.findall(r"\w+", pool, flags=re.UNICODE))
        score = len(token_pool.intersection(terms))

        if "objective" in pool:
            score += 1
        if "materials" in pool:
            score += 1
        if "procedure" in pool:
            score += 1
        if title.lower() in {"general", "overview"}:
            score -= 1
        return score

    return max(blocks, key=score_block)


def _extract_labeled_items(text: str, label: str) -> list[str]:
    target = label.lower()
    items: list[str] = []
    collecting = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("### "):
            if collecting:
                break
            continue

        match = re.match(r"^([A-Za-z][A-Za-z0-9 /()_-]+):\s*(.*)$", line)
        if match:
            current_label = match.group(1).lower()
            trailing = _clean_markdown_line(match.group(2))

            if collecting and current_label != target:
                break

            if current_label == target:
                collecting = True
                if trailing:
                    items.append(trailing)
                continue

        if collecting:
            cleaned = _clean_markdown_line(line)
            if cleaned:
                items.append(cleaned)

    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _extract_observation_items(text: str) -> list[str]:
    observations: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith("visual:") or lowered.startswith("- visual:"):
            observations.append(_clean_markdown_line(line))
        elif lowered.startswith("data cue:") or lowered.startswith("- data cue:"):
            observations.append(_clean_markdown_line(line))

    if not observations:
        observations = _extract_labeled_items(text, "Expected comparative visuals/data")

    deduped: list[str] = []
    for item in observations:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _extract_key_facts(section_text: str, *, max_items: int = 2) -> list[str]:
    facts: list[str] = []

    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if line.startswith("- "):
            cleaned = _clean_markdown_line(line)
            if cleaned:
                facts.append(cleaned)
        if len(facts) >= max_items:
            break

    if not facts:
        facts = _extract_brief_facts(section_text, max_sentences=max_items)

    deduped: list[str] = []
    for fact in facts:
        if fact not in deduped:
            deduped.append(fact)
    return deduped[:max_items]


def _collect_source_sections(retrieval: Any, database_dir: Path) -> dict[tuple[str, str], str]:
    source_sections: dict[tuple[str, str], str] = {}
    cached_files: dict[str, dict[str, str]] = {}

    for chunk in retrieval.chunks:
        key = (chunk.source_file, chunk.section)
        if key in source_sections:
            continue

        source_path = database_dir / chunk.source_file
        section_text = ""

        if source_path.exists():
            if chunk.source_file not in cached_files:
                cached_files[chunk.source_file] = _split_h2_sections(
                    source_path.read_text(encoding="utf-8")
                )

            section_text = cached_files[chunk.source_file].get(chunk.section.lower().strip(), "")

        source_sections[key] = section_text or chunk.content

    return source_sections


def _build_script_lines(query: str, retrieval: Any, database_dir: Path) -> list[str]:
    section_map = _collect_source_sections(retrieval, database_dir)
    section_items = list(section_map.items())

    simulator_entry = next(
        (
            item
            for item in section_items
            if "simulator-ready experiment data" in item[0][1]
        ),
        section_items[0],
    )

    simulator_title, simulator_block = _select_best_h3_block(simulator_entry[1], query)

    objective_items = _extract_labeled_items(simulator_block, "Objective")
    material_items = _extract_labeled_items(simulator_block, "Materials")
    control_items = _extract_labeled_items(simulator_block, "Control parameters")

    procedure_items = _extract_labeled_items(simulator_block, "Procedure and expected visuals")
    if not procedure_items:
        procedure_items = _extract_labeled_items(simulator_block, "Expected comparative visuals/data")

    procedure_steps = [
        item
        for item in procedure_items
        if not item.lower().startswith("visual:") and not item.lower().startswith("data cue:")
    ]

    observation_items = _extract_observation_items(simulator_block)

    safety_items: list[str] = []
    for _, section_text in section_items:
        for item in _extract_labeled_items(section_text, "Safety notes"):
            if item not in safety_items:
                safety_items.append(item)

    support_facts: list[str] = []
    for (source_file, section_name), section_text in section_items:
        if (source_file, section_name) == simulator_entry[0]:
            continue
        for fact in _extract_key_facts(section_text, max_items=2):
            if fact not in support_facts:
                support_facts.append(fact)
        if len(support_facts) >= 4:
            break

    lines: list[str] = []
    lines.append("## Grounded Scripting Script")
    lines.append("### Script title")
    if simulator_title and simulator_title.lower() not in {"general", "overview"}:
        lines.append(f"- {simulator_title}")
    else:
        lines.append("- Grounded experiment script")

    lines.append("")
    lines.append("### Experiment objective")
    if objective_items:
        for item in objective_items[:2]:
            lines.append(f"- {item}")
    else:
        lines.append(f"- {query}")

    lines.append("")
    lines.append("### Materials")
    if material_items:
        for item in material_items[:8]:
            lines.append(f"- {item}")
    else:
        lines.append("- Use materials from retrieved chunk references below.")

    lines.append("")
    lines.append("### Setup and control parameters")
    if control_items:
        for item in control_items[:8]:
            lines.append(f"- {item}")
    else:
        lines.append("- Apply constraints grounded in retrieved chunk references.")

    lines.append("")
    lines.append("### Procedure")
    if procedure_steps:
        for idx, step in enumerate(procedure_steps[:8], start=1):
            lines.append(f"{idx}. {step}")
    else:
        lines.append("1. Follow the simulator-ready steps grounded in retrieved chunks.")

    lines.append("")
    lines.append("### Expected observations")
    if observation_items:
        for item in observation_items[:8]:
            lines.append(f"- {item}")
    elif support_facts:
        for fact in support_facts[:4]:
            lines.append(f"- {fact}")
    else:
        lines.append("- Observe effects exactly as stated in grounded references.")

    lines.append("")
    lines.append("### Safety constraints")
    if safety_items:
        for item in safety_items[:6]:
            lines.append(f"- {item}")
    else:
        lines.append("- Follow safety notes from retrieved chunk references.")

    return lines


def _build_chunk_lines(retrieval: Any) -> list[str]:
    lines = ["## Retrieved Chunks"]

    def to_cosine_similarity(distance: float) -> float:
        # Chroma returns cosine distance when hnsw:space is "cosine".
        # Convert to similarity for easier interpretation: similarity = 1 - distance.
        return max(-1.0, min(1.0, 1.0 - float(distance)))

    for idx, chunk in enumerate(retrieval.chunks, start=1):
        cosine_similarity = to_cosine_similarity(chunk.distance)
        lines.append(
            f"- Chunk {idx}: topic={chunk.topic}, source={chunk.source_file}, "
            f"section={chunk.section}, cosine_similarity={cosine_similarity:.4f}, "
            f"chunk_id={chunk.chunk_id}"
        )
    return lines


def _render_grounded_output(query: str, retrieval: Any, *, database_dir: Path) -> str:
    if not retrieval.has_enough_data:
        return "I don't have enough scientific data"

    lines: list[str] = []
    lines.append("# Scripting RAG Demo Output")
    lines.append("")
    lines.append("## Teacher Prompt")
    lines.append(query)
    lines.append("")
    lines.extend(_build_script_lines(query, retrieval, database_dir))
    lines.append("")
    lines.extend(_build_chunk_lines(retrieval))
    return "\n".join(lines)


def run_demo(query: str, *, top_k: int = 4, database_dir: str | None = None) -> str:
    retriever_module = _load_retriever_module()
    _ensure_local_backends(retriever_module)

    ScriptingScienceRetriever = retriever_module.ScriptingScienceRetriever
    repo_root = Path(__file__).resolve().parents[3]
    script_root = Path(__file__).resolve().parent
    resolved_database_dir = Path(database_dir) if database_dir else (repo_root / "mock_science_db")

    retriever = ScriptingScienceRetriever(
        database_dir=resolved_database_dir,
        persist_dir=script_root / ".demo_rag_index",
        embedding_client=_KeywordEmbeddings(),
        default_top_k=max(1, min(top_k, 8)),
    )

    retrieval = retriever.retrieve(query, top_k=max(1, min(top_k, 8)))
    return _render_grounded_output(query, retrieval, database_dir=resolved_database_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demo prompt -> RAG retrieval -> grounded scripting output",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Teacher prompt for scripting agent (omit to use interactive prompt)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Always prompt in terminal for 'Teacher prompt'",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Maximum chunks to retrieve (default: 4)",
    )
    parser.add_argument(
        "--database-dir",
        default=None,
        help="Path to markdown science database (default: <repo>/mock_science_db)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    query = args.query
    if args.interactive or not query:
        query = input("Teacher prompt: ").strip()

    if not query:
        print("Teacher prompt is empty. Please run again and enter a prompt.")
        return

    output = run_demo(query, top_k=args.top_k, database_dir=args.database_dir)
    print(output)


if __name__ == "__main__":
    main()
