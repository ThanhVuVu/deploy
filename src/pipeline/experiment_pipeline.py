"""
End-to-end teacher prompt -> RAG -> scripting -> simulator pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from src.agents.scripting.scripting_agent import ScriptingAgent
from src.agents.scripting.rag import MultimodalRetriever, ScriptingScienceRetriever
from src.agents.simulator import SimulatorAgent, SimulatorArtifacts, SimulatorOutputError


class _ScriptingLike(Protocol):
    def run(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
        messages: Optional[list[Any]] = None,
    ) -> str:
        ...


class _SimulatorLike(Protocol):
    def generate(
        self,
        *,
        teacher_prompt: str,
        experiment_script: str,
        retrieved_context: str = "",
        retrieved_sources: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> SimulatorArtifacts:
        ...


@dataclass(frozen=True)
class RAGSnapshot:
    """Compact RAG output shape used by both downstream agents."""

    has_enough_data: bool
    grounded_context: str
    source_report: str


@dataclass(frozen=True)
class ExperimentPipelineResult:
    """End-to-end generation result."""

    teacher_prompt: str
    experiment_script: str
    rag_snapshot: RAGSnapshot
    artifacts: SimulatorArtifacts
    output_dir: Optional[Path] = None
    written_files: tuple[Path, ...] = ()


class ExperimentGenerationPipeline:
    """
    Orchestrates the full MVP flow:

    teacher prompt -> RAG retrieval -> scripting agent -> simulator agent -> files.
    """

    def __init__(
        self,
        *,
        scripting_agent: Optional[_ScriptingLike] = None,
        simulator_agent: Optional[_SimulatorLike] = None,
        retriever: Optional[Any] = None,
        use_multimodal: bool = True,
    ) -> None:
        self.retriever = retriever or (
            MultimodalRetriever() if use_multimodal else ScriptingScienceRetriever()
        )
        self.scripting_agent = scripting_agent or ScriptingAgent(
            use_multimodal=use_multimodal,
            multimodal_retriever=self.retriever if use_multimodal else None,
            retriever=self.retriever if not use_multimodal else None,
        )
        self.simulator_agent = simulator_agent or SimulatorAgent()

    def run(
        self,
        teacher_prompt: str,
        *,
        output_dir: str | Path | None = None,
        output_filename: str | None = None,
        rag_top_k: int = 4,
        context: Optional[dict[str, Any]] = None,
        write_files: bool = True,
    ) -> ExperimentPipelineResult:
        prompt = teacher_prompt.strip()
        if not prompt:
            raise ValueError("teacher_prompt must not be empty.")

        context = {**(context or {}), "rag_top_k": rag_top_k}
        rag_snapshot = self.retrieve(prompt, top_k=rag_top_k)

        if not rag_snapshot.has_enough_data:
            raise SimulatorOutputError(
                "RAG did not return enough grounded data for this prompt."
            )

        experiment_script = self.scripting_agent.run(prompt, context=context)
        if not experiment_script.strip() or experiment_script == ScriptingAgent._NO_DATA_MESSAGE:
            raise SimulatorOutputError(
                "ScriptingAgent did not produce a grounded experiment script."
            )

        artifacts = self.simulator_agent.generate(
            teacher_prompt=prompt,
            experiment_script=experiment_script,
            retrieved_context=rag_snapshot.grounded_context,
            retrieved_sources=rag_snapshot.source_report,
            context=context,
        )

        written: tuple[Path, ...] = ()
        resolved_output_dir: Optional[Path] = Path(output_dir) if output_dir else None
        if write_files and resolved_output_dir is not None:
            written = tuple(
                artifacts.write_to(
                    resolved_output_dir,
                    filename=output_filename or "experiment.html",
                )
            )

        return ExperimentPipelineResult(
            teacher_prompt=prompt,
            experiment_script=experiment_script,
            rag_snapshot=rag_snapshot,
            artifacts=artifacts,
            output_dir=resolved_output_dir,
            written_files=written,
        )

    def retrieve(self, teacher_prompt: str, *, top_k: int = 4) -> RAGSnapshot:
        """Run whichever retriever shape is configured and normalize the result."""
        if hasattr(self.retriever, "hybrid_retrieve"):
            result = self.retriever.hybrid_retrieve(teacher_prompt, top_k=top_k)
        else:
            result = self.retriever.retrieve(teacher_prompt, top_k=top_k)

        return RAGSnapshot(
            has_enough_data=bool(getattr(result, "has_enough_data", False)),
            grounded_context=str(getattr(result, "grounded_context", "")),
            source_report=str(getattr(result, "source_report", "")),
        )
