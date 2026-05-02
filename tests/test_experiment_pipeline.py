from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.agents.simulator import SimulatorArtifacts, SimulatorOutputError
from src.pipeline.experiment_pipeline import ExperimentGenerationPipeline


class _FakeRetrieval:
    has_enough_data = True
    grounded_context = "Retrieved fact: water boils at 100 C."
    source_report = "- mock.md | page=1"


class _EmptyRetrieval:
    has_enough_data = False
    grounded_context = ""
    source_report = ""


class _FakeRetriever:
    def __init__(self, result) -> None:
        self.result = result
        self.last_query = None
        self.last_top_k = None

    def hybrid_retrieve(self, query: str, top_k: int = 4):
        self.last_query = query
        self.last_top_k = top_k
        return self.result


class _FakeScriptingAgent:
    def __init__(self, script: str = "Grounded script.") -> None:
        self.script = script
        self.last_context = None

    def run(self, query, context=None, messages=None):
        self.last_context = context
        return self.script


@dataclass
class _FakeSimulatorAgent:
    last_payload: dict | None = None

    def generate(
        self,
        *,
        teacher_prompt,
        experiment_script,
        retrieved_context="",
        retrieved_sources="",
        context=None,
    ):
        self.last_payload = {
            "teacher_prompt": teacher_prompt,
            "experiment_script": experiment_script,
            "retrieved_context": retrieved_context,
            "retrieved_sources": retrieved_sources,
            "context": context,
        }
        return SimulatorArtifacts(
            structured_understanding="facts",
            dsl="dsl",
            architecture="arch",
            index_html="<html><head><script src='p5.js'></script><script src='sketch.js'></script></head></html>",
            sketch_js="function setup(){createCanvas(10,10);} function draw(){}",
        )


def test_pipeline_passes_rag_and_script_to_simulator():
    retriever = _FakeRetriever(_FakeRetrieval())
    scripting = _FakeScriptingAgent("Script from scripting agent.")
    simulator = _FakeSimulatorAgent()
    pipeline = ExperimentGenerationPipeline(
        retriever=retriever,
        scripting_agent=scripting,
        simulator_agent=simulator,
    )

    result = pipeline.run(
        "Teacher prompt",
        rag_top_k=5,
        output_dir="generated_test",
        write_files=False,
    )

    assert retriever.last_query == "Teacher prompt"
    assert retriever.last_top_k == 5
    assert scripting.last_context["rag_top_k"] == 5
    assert simulator.last_payload["experiment_script"] == "Script from scripting agent."
    assert "water boils" in simulator.last_payload["retrieved_context"]
    assert result.output_dir.name == "generated_test"
    assert result.written_files == ()


def test_pipeline_writes_single_named_html_file():
    pipeline = ExperimentGenerationPipeline(
        retriever=_FakeRetriever(_FakeRetrieval()),
        scripting_agent=_FakeScriptingAgent(),
        simulator_agent=_FakeSimulatorAgent(),
    )
    output_dir = Path(f"generated_test_pipeline_{uuid.uuid4().hex}")

    result = pipeline.run(
        "Teacher prompt",
        output_dir=output_dir,
        output_filename="teacher-prompt-120000-20260502.html",
        write_files=True,
    )

    assert len(result.written_files) == 1
    assert result.written_files[0].name == "teacher-prompt-120000-20260502.html"
    assert result.written_files[0].read_text(encoding="utf-8").count("<script") >= 2


def test_pipeline_rejects_empty_rag():
    pipeline = ExperimentGenerationPipeline(
        retriever=_FakeRetriever(_EmptyRetrieval()),
        scripting_agent=_FakeScriptingAgent(),
        simulator_agent=_FakeSimulatorAgent(),
    )

    with pytest.raises(SimulatorOutputError, match="RAG"):
        pipeline.run("Out of scope")
