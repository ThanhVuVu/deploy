"""
CLI for running only the simulator stage from an existing scripting output.

Usage:
    python -m src.agents.simulator.cli --script script.md --out-dir generated/demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agents.simulator import SimulatorAgent, SimulatorOutputError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one standalone HTML simulator file from a grounded script."
    )
    parser.add_argument("--script", required=True, help="Grounded scripting output file.")
    parser.add_argument(
        "--teacher-prompt",
        default="",
        help="Original teacher prompt, if available.",
    )
    parser.add_argument(
        "--rag-context",
        default=None,
        help="Optional file containing retrieved RAG context.",
    )
    parser.add_argument(
        "--rag-sources",
        default=None,
        help="Optional file containing RAG source trace.",
    )
    parser.add_argument(
        "--out-dir",
        default="generated_experiment",
        help="Directory for the generated standalone HTML file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override simulator LLM model, e.g. gpt-5.5.",
    )
    return parser


def _read_optional(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8").strip()


def main() -> int:
    load_dotenv()
    args = _build_parser().parse_args()

    script = Path(args.script).read_text(encoding="utf-8").strip()
    agent = SimulatorAgent(llm_model=args.model)

    try:
        artifacts = agent.generate(
            teacher_prompt=args.teacher_prompt or "Generate the requested virtual lab.",
            experiment_script=script,
            retrieved_context=_read_optional(args.rag_context),
            retrieved_sources=_read_optional(args.rag_sources),
        )
        written = artifacts.write_to(args.out_dir)
    except SimulatorOutputError as exc:
        print(f"Simulator generation failed: {exc}", file=sys.stderr)
        return 2

    print("Simulator files written:")
    for path in written:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
