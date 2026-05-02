from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agents.simulator import (
    SimulatorAgent,
    SimulatorOutputError,
    SimulatorOutputParser,
)


def _valid_js() -> str:
    return """
let temperature = 25;

function setup() {
  createCanvas(640, 360);
}

function draw() {
  background(20);
  text(temperature, 20, 20);
}
"""


def _valid_html() -> str:
    return """<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/p5@1.9.4/lib/p5.min.js"></script>
  <script src="sketch.js"></script>
</head>
<body></body>
</html>
"""


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = None

    def chat(self, messages, **kwargs):
        self.messages = messages
        return SimpleNamespace(content=self.content)


class TestSimulatorOutputParser:
    def test_parse_json_payload_with_files(self):
        raw = json.dumps(
            {
                "structured_understanding": "facts",
                "dsl": {"entities": ["water"]},
                "architecture": "modules",
                "files": {
                    "index.html": _valid_html(),
                    "sketch.js": _valid_js(),
                },
                "validation_checklist": ["uses p5"],
            }
        )

        artifacts = SimulatorOutputParser().parse(raw)

        assert artifacts.structured_understanding == "facts"
        assert "water" in artifacts.dsl
        assert artifacts.index_html.strip() == _valid_html().strip()
        assert artifacts.sketch_js.strip() == _valid_js().strip()
        assert artifacts.validation_checklist == ("uses p5",)

    def test_parse_fenced_blocks(self):
        raw = f"""Structured understanding: demo

```html
{_valid_html()}
```

```javascript
{_valid_js()}
```
"""

        artifacts = SimulatorOutputParser().parse(raw)

        assert "sketch.js" in artifacts.index_html
        assert "function setup" in artifacts.sketch_js

    def test_missing_draw_raises(self):
        raw = json.dumps(
            {
                "files": {
                    "index.html": _valid_html(),
                    "sketch.js": "function setup() { createCanvas(10, 10); }",
                }
            }
        )

        with pytest.raises(SimulatorOutputError, match="draw"):
            SimulatorOutputParser().parse(raw)

    def test_default_html_added_when_only_js_exists(self):
        raw = f"```javascript\n{_valid_js()}\n```"

        artifacts = SimulatorOutputParser().parse(raw)

        assert "p5" in artifacts.index_html.lower()
        assert "sketch.js" in artifacts.index_html


class TestSimulatorAgent:
    def test_gpt5_simulator_uses_default_temperature(self, monkeypatch):
        captured = {}

        def fake_get_provider(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

        monkeypatch.setenv("SIMULATOR_PROVIDER_BACKEND", "openai")
        monkeypatch.setenv("SIMULATOR_LLM_MODEL", "gpt-5.5")
        monkeypatch.setattr(
            "src.agents.simulator.simulator_agent.get_provider",
            fake_get_provider,
        )
        monkeypatch.setattr(
            "src.agents.simulator.simulator_agent.OpenAIClient",
            MagicMock(),
        )

        SimulatorAgent(temperature=0.1)

        assert captured["model"] == "gpt-5.5"
        assert captured["temperature"] == 1.0

    def test_non_gpt5_simulator_keeps_requested_temperature(self, monkeypatch):
        captured = {}

        def fake_get_provider(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

        monkeypatch.setenv("SIMULATOR_PROVIDER_BACKEND", "openai")
        monkeypatch.setenv("SIMULATOR_LLM_MODEL", "gpt-4o")
        monkeypatch.setattr(
            "src.agents.simulator.simulator_agent.get_provider",
            fake_get_provider,
        )
        monkeypatch.setattr(
            "src.agents.simulator.simulator_agent.OpenAIClient",
            MagicMock(),
        )

        SimulatorAgent(temperature=0.1)

        assert captured["model"] == "gpt-4o"
        assert captured["temperature"] == 0.1

    def test_generate_sends_script_and_rag_to_client(self):
        payload = json.dumps(
            {
                "structured_understanding": "facts",
                "dsl": "dsl",
                "architecture": "architecture",
                "files": {
                    "index.html": _valid_html(),
                    "sketch.js": _valid_js(),
                },
            }
        )
        client = _FakeClient(payload)
        agent = SimulatorAgent(client=client)

        artifacts = agent.generate(
            teacher_prompt="Compare boiling points.",
            experiment_script="Script with boiling data.",
            retrieved_context="oxygen boils at -183 C",
            retrieved_sources="- source page 1",
        )

        assert "function draw" in artifacts.sketch_js
        user_prompt = client.messages[1]["content"]
        assert "Compare boiling points." in user_prompt
        assert "Script with boiling data." in user_prompt
        assert "oxygen boils at -183 C" in user_prompt
        assert "- source page 1" in user_prompt

    def test_generate_requires_script(self):
        agent = SimulatorAgent(client=_FakeClient("{}"))

        with pytest.raises(SimulatorOutputError, match="requires"):
            agent.generate(teacher_prompt="Prompt", experiment_script="")
