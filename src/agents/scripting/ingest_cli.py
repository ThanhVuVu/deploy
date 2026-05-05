"""
src/agents/scripting/ingest_cli.py

CLI tool to ingest PDF files from science_db/ into the multimodal ChromaDB index.

Usage
-----
# Ingest a specific PDF:
    python -m src.agents.scripting.ingest_cli --pdf science_db/science_db_6.pdf

# Ingest all PDFs in science_db/:
    python -m src.agents.scripting.ingest_cli --all

# Show current index stats (no ingestion):
    python -m src.agents.scripting.ingest_cli --stats

# Run a retrieval query to verify:
    python -m src.agents.scripting.ingest_cli --query "thí nghiệm nam châm"

Notes
-----
* Ingestion is idempotent: already-ingested PDFs are detected via SHA256 manifest
  and skipped entirely. Safe to run multiple times.
* OPENAI_API_KEY must be set in .env or environment.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Ensure project root is on sys.path when run as __main__ ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # …/A20-App-115
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env (OPENAI_API_KEY, etc.)
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_cli")

DEFAULT_SCIENCE_DB = _PROJECT_ROOT / "science_db"
DEFAULT_PERSIST_DIR = _PROJECT_ROOT / ".rag" / "multimodal_chroma"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multimodal RAG ingest tool for Virtual Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pdf",
        metavar="PATH",
        help="Path to a single PDF file to ingest.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help=f"Ingest all PDF and JSON files found in --db-dir (default: {DEFAULT_SCIENCE_DB}).",
    )
    group.add_argument(
        "--json",
        metavar="PATH",
        help="Path to a single JSON file to ingest.",
    )
    group.add_argument(
        "--stats",
        action="store_true",
        help="Print current index collection counts and exit.",
    )
    group.add_argument(
        "--all-json",
        action="store_true",
        help=f"Ingest all JSON files found in --db-dir (default: {DEFAULT_SCIENCE_DB}), skipping PDFs.",
    )
    group.add_argument(
        "--query",
        metavar="TEXT",
        help="Run a hybrid retrieval query and print top results.",
    )
    p.add_argument(
        "--db-dir",
        metavar="DIR",
        default=str(DEFAULT_SCIENCE_DB),
        help=f"Source directory for PDF/JSON files (default: {DEFAULT_SCIENCE_DB}).",
    )
    p.add_argument(
        "--persist-dir",
        metavar="DIR",
        default=str(DEFAULT_PERSIST_DIR),
        help=f"ChromaDB persistence dir (default: {DEFAULT_PERSIST_DIR}).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of results for --query mode (default: 4).",
    )
    return p


def cmd_stats(indexer) -> None:
    counts = indexer.collection_counts()
    print("\n[STATS] Index collection counts:")
    for name, count in counts.items():
        print(f"   {name:<25} {count:>6} chunks")
    print()


def cmd_ingest_pdf(indexer, pdf_path: Path) -> None:
    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        sys.exit(1)

    print(f"\n[INGEST] Ingesting: {pdf_path.name} …")
    stats = indexer.ingest_pdf(pdf_path)
    print(f"[OK] {stats}")
    if not stats.skipped:
        print(
            f"   text={stats.texts_added}  image={stats.images_added}  "
            f"table={stats.tables_added}  total={stats.total_added}"
        )


def cmd_ingest_json(indexer, json_path: Path) -> None:
    if not json_path.exists():
        logger.error("File not found: %s", json_path)
        sys.exit(1)

    print(f"\n[INGEST] Ingesting JSON: {json_path.name} …")
    stats = indexer.ingest_json(json_path)
    print(f"[OK] {stats}")
    if not stats.skipped:
        print(
            f"   text={stats.texts_added}  image={stats.images_added}  "
            f"table={stats.tables_added}  total={stats.total_added}"
        )


def cmd_ingest_all(indexer, db_dir: Path) -> None:
    pdfs = sorted(db_dir.glob("*.pdf"))
    jsons = sorted(db_dir.glob("*.json"))
    
    if not pdfs and not jsons:
        logger.warning("No PDF or JSON files found in %s", db_dir)
        return

    print(f"\n[DIR] Found {len(pdfs)} PDF(s) and {len(jsons)} JSON(s) in {db_dir}")
    for pdf in pdfs:
        cmd_ingest_pdf(indexer, pdf)
    for j in jsons:
        cmd_ingest_json(indexer, j)

    print("\n[STATS] Final index counts:")
    cmd_stats(indexer)


def cmd_ingest_all_json(indexer, db_dir: Path) -> None:
    jsons = sorted(db_dir.glob("*.json"))
    if not jsons:
        logger.warning("No JSON files found in %s", db_dir)
        return

    print(f"\n[DIR] Found {len(jsons)} JSON(s) in {db_dir}")
    for j in jsons:
        cmd_ingest_json(indexer, j)

    print("\n[STATS] Final index counts:")
    cmd_stats(indexer)


def cmd_query(retriever, query: str, top_k: int) -> None:
    print(f"\n[QUERY] Query: {query!r}  (top_k={top_k})\n")
    result = retriever.hybrid_retrieve(query, top_k=top_k)

    if not result.has_enough_data:
        print("[WARNING] No relevant results found.")
        return

    print("── Source trace ─────────────────────────────────────────────────")
    print(result.source_report)
    print("\n── Grounded context ──────────────────────────────────────────────")
    print(result.grounded_context)
    print()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Lazy import after dotenv is loaded
    from src.agents.scripting.rag import MultimodalIndexer, MultimodalRetriever

    persist_dir = args.persist_dir

    # ── Stats only: lightweight, no embedders needed ───────────────────────
    if args.stats:
        indexer = MultimodalIndexer(persist_dir=persist_dir)
        cmd_stats(indexer)
        return

    # ── Query mode ─────────────────────────────────────────────────────────
    if args.query:
        indexer = MultimodalIndexer(persist_dir=persist_dir)
        cols = indexer.get_collections()
        if all(c.count() == 0 for c in cols):
            print("⚠️   Index is empty. Run --all or --pdf first.")
            sys.exit(1)
        col_text, col_image, col_table = cols
        retriever = MultimodalRetriever(
            col_text=col_text,
            col_image=col_image,
            col_table=col_table,
        )
        cmd_query(retriever, args.query, args.top_k)
        return

    # ── Ingest mode ────────────────────────────────────────────────────────
    indexer = MultimodalIndexer(persist_dir=persist_dir)

    if args.all:
        cmd_ingest_all(indexer, Path(args.db_dir))
    elif args.all_json:
        cmd_ingest_all_json(indexer, Path(args.db_dir))
    elif args.pdf:
        cmd_ingest_pdf(indexer, Path(args.pdf))
    elif args.json:
        cmd_ingest_json(indexer, Path(args.json))


if __name__ == "__main__":
    main()
