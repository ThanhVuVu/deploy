"""
CLI for the full Virtual Lab generation pipeline.

Usage:
    python -m src.pipeline.generate_experiment_cli "Thí nghiệm ..." --out-dir generated/demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agents.scripting.scripting_agent import ScriptingAgent
from src.agents.scripting.rag import MultimodalRetriever, ScriptingScienceRetriever
from src.agents.simulator import SimulatorAgent, SimulatorOutputError
from src.pipeline.experiment_pipeline import ExperimentGenerationPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate grounded p5.js simulation files from a teacher prompt."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Teacher prompt. If omitted, use --prompt-file.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Read teacher prompt from a UTF-8 text file.",
    )
    parser.add_argument(
        "--out-dir",
        default="generated_experiment",
        help="Directory for index.html and sketch.js (default: generated_experiment).",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=6,
        help="Number of RAG chunks to pass through the pipeline (default: 6).",
    )
    parser.add_argument(
        "--simulator-model",
        default=None,
        help="Override simulator LLM model, e.g. gpt-5.5.",
    )
    parser.add_argument(
        "--text-rag",
        action="store_true",
        help="Use legacy text-only RAG instead of multimodal RAG.",
    )
    return parser


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    return (args.prompt or "").strip()


def main() -> int:
    load_dotenv()

    parser = _build_parser()
    args = parser.parse_args()
    prompt = _read_prompt(args)
    if not prompt:
        parser.error("Provide a prompt argument or --prompt-file.")

    use_multimodal = not args.text_rag
    retriever = MultimodalRetriever() if use_multimodal else ScriptingScienceRetriever()
    scripting_agent = ScriptingAgent(
        use_multimodal=use_multimodal,
        multimodal_retriever=retriever if use_multimodal else None,
        retriever=retriever if not use_multimodal else None,
    )
    simulator_agent = SimulatorAgent(llm_model=args.simulator_model)
    pipeline = ExperimentGenerationPipeline(
        retriever=retriever,
        scripting_agent=scripting_agent,
        simulator_agent=simulator_agent,
        use_multimodal=use_multimodal,
    )

    try:
        result = pipeline.run(
            prompt,
            output_dir=args.out_dir,
            rag_top_k=args.rag_top_k,
            write_files=True,
        )
    except SimulatorOutputError as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 2

    print("Generation complete.")
    print(f"Output directory: {result.output_dir}")
    for path in result.written_files:
        print(f"- {path}")
    print("\nRAG sources:")
    print(result.rag_snapshot.source_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
