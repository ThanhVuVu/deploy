"""
Simulator agent for turning grounded experiment scripts into p5.js files.

The scripting agent produces the educational plan.  This agent is responsible
for producing executable simulator artifacts while preserving the RAG grounding
that produced the plan.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from langchain_core.messages import BaseMessage

from src.agents.base.base_agent import BaseAgent
from src.llm.openai_client import OpenAIClient
from src.llm.output_parser import OutputParser
from src.llm.provider import get_provider

logger = logging.getLogger(__name__)


class _ChatClient(Protocol):
    """Small protocol for OpenAIClient-compatible test doubles."""

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        ...


class SimulatorOutputError(ValueError):
    """Raised when simulator LLM output cannot be parsed into runnable files."""


@dataclass(frozen=True)
class SimulatorArtifacts:
    """Parsed simulator output files plus design trace sections."""

    structured_understanding: str
    dsl: str
    architecture: str
    index_html: str
    sketch_js: str
    experiment_html: str = ""
    raw_output: str = ""
    validation_checklist: tuple[str, ...] = field(default_factory=tuple)

    @property
    def files(self) -> dict[str, str]:
        return {"experiment.html": self.single_file_html}

    @property
    def single_file_html(self) -> str:
        """Return the runnable one-file HTML artifact."""
        if self.experiment_html.strip():
            return self.experiment_html.strip()
        return self._inline_sketch(self.index_html, self.sketch_js)

    def write_to(self, output_dir: str | Path, filename: str = "experiment.html") -> list[Path]:
        """Write the runnable experiment as a single HTML file."""
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        safe_filename = Path(filename).name or "experiment.html"
        if not safe_filename.lower().endswith(".html"):
            safe_filename += ".html"

        path = target / safe_filename
        path.write_text(self.single_file_html, encoding="utf-8")
        return [path]

    @staticmethod
    def _inline_sketch(index_html: str, sketch_js: str) -> str:
        """Inline sketch.js into index.html for standalone browser execution."""
        html = (index_html or "").strip()
        sketch = (sketch_js or "").strip()

        if not html:
            html = SimulatorOutputParser.default_index_html()

        if sketch:
            inline_script = f"<script>\n{sketch}\n</script>"
            replaced = re.sub(
                r"<script[^>]+src=[\"']sketch\.js[\"'][^>]*>\s*</script>",
                inline_script,
                html,
                flags=re.IGNORECASE,
            )
            if replaced == html:
                if "</body>" in html.lower():
                    replaced = re.sub(
                        r"</body>",
                        inline_script + "\n</body>",
                        html,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                else:
                    replaced = html + "\n" + inline_script
            html = replaced

        if "p5" not in html.lower():
            html = SimulatorOutputParser._insert_before(
                html,
                "</head>",
                '  <script src="https://cdn.jsdelivr.net/npm/p5@1.9.4/lib/p5.min.js"></script>\n',
            )
        return html


class SimulatorOutputParser:
    """Parse and validate simulator-agent output."""

    _SETUP_PATTERN = re.compile(
        r"(\bfunction\s+setup\s*\(|\bp\.setup\s*=|setup\s*:\s*function)",
        re.IGNORECASE,
    )
    _DRAW_PATTERN = re.compile(
        r"(\bfunction\s+draw\s*\(|\bp\.draw\s*=|draw\s*:\s*function)",
        re.IGNORECASE,
    )

    def parse(self, raw_output: str) -> SimulatorArtifacts:
        if not raw_output or not raw_output.strip():
            raise SimulatorOutputError("Simulator output is empty.")

        parsed = OutputParser.parse(raw_output)
        payload = self._load_json_payload(raw_output)
        if payload is None and isinstance(parsed.json, dict):
            payload = parsed.json

        structured_understanding = ""
        dsl = ""
        architecture = ""
        experiment_html = ""
        index_html = ""
        sketch_js = ""
        validation: tuple[str, ...] = ()

        if payload:
            structured_understanding = self._stringify(
                payload.get("structured_understanding")
                or payload.get("understanding")
                or payload.get("structuredUnderstanding")
            )
            dsl = self._stringify(payload.get("dsl") or payload.get("DSL"))
            architecture = self._stringify(
                payload.get("architecture") or payload.get("architecture_explanation")
            )
            validation = tuple(
                str(item)
                for item in (
                    payload.get("validation_checklist")
                    or payload.get("validation")
                    or []
                )
            )

            files = payload.get("files") if isinstance(payload.get("files"), dict) else {}
            experiment_html = self._stringify(
                files.get("experiment.html")
                or files.get("experiment_html")
                or payload.get("experiment_html")
                or payload.get("html")
            )
            index_html = self._stringify(
                files.get("index.html")
                or files.get("index_html")
                or payload.get("index_html")
            )
            sketch_js = self._stringify(
                files.get("sketch.js")
                or files.get("sketch_js")
                or files.get("javascript")
                or payload.get("sketch_js")
                or payload.get("javascript")
                or payload.get("js")
            )

        experiment_html = self._strip_outer_fence(experiment_html)
        if not index_html:
            index_html = (
                parsed.first_block("html")
                or self._extract_named_block(raw_output, "index.html")
                or ""
            )
        if not sketch_js:
            sketch_js = (
                parsed.first_block("javascript")
                or parsed.first_block("js")
                or self._extract_named_block(raw_output, "sketch.js")
                or ""
            )

        index_html = self._strip_outer_fence(index_html)
        sketch_js = self._strip_outer_fence(sketch_js)

        if experiment_html:
            if not self._extract_inline_script(experiment_html) and sketch_js:
                experiment_html = SimulatorArtifacts._inline_sketch(
                    experiment_html,
                    sketch_js,
                )
            index_html = self._ensure_p5_html(experiment_html)
            sketch_js = self._extract_inline_script(index_html)
        elif not sketch_js and index_html:
            inline_script = self._extract_inline_script(index_html)
            if inline_script:
                sketch_js = inline_script
                experiment_html = self._ensure_p5_html(index_html)
                index_html = self.default_index_html()

        if not index_html:
            index_html = self.default_index_html()

        index_html = self._ensure_p5_index(index_html)
        if not experiment_html:
            experiment_html = SimulatorArtifacts._inline_sketch(index_html, sketch_js)

        self.validate(index_html=index_html, sketch_js=sketch_js)
        self.validate_experiment_html(experiment_html)

        return SimulatorArtifacts(
            structured_understanding=structured_understanding,
            dsl=dsl,
            architecture=architecture,
            index_html=index_html,
            sketch_js=sketch_js,
            experiment_html=experiment_html,
            raw_output=raw_output,
            validation_checklist=validation,
        )

    def validate(self, *, index_html: str, sketch_js: str) -> None:
        """Validate minimum runnable p5.js artifact requirements."""
        if not sketch_js or not sketch_js.strip():
            raise SimulatorOutputError("Missing sketch.js content.")
        if not self._SETUP_PATTERN.search(sketch_js):
            raise SimulatorOutputError("sketch.js must define p5 setup().")
        if not self._DRAW_PATTERN.search(sketch_js):
            raise SimulatorOutputError("sketch.js must define p5 draw().")
        if "createCanvas" not in sketch_js and "createcanvas" not in sketch_js.lower():
            raise SimulatorOutputError("sketch.js must create a p5 canvas.")
        if "sketch.js" not in index_html:
            raise SimulatorOutputError("index.html must load sketch.js.")
        if "p5" not in index_html.lower():
            raise SimulatorOutputError("index.html must load p5.js.")

    def validate_experiment_html(self, experiment_html: str) -> None:
        """Validate the single-file runnable HTML artifact."""
        if not experiment_html or not experiment_html.strip():
            raise SimulatorOutputError("Missing experiment.html content.")
        if "p5" not in experiment_html.lower():
            raise SimulatorOutputError("experiment.html must load p5.js.")
        inline_script = self._extract_inline_script(experiment_html)
        if not inline_script:
            raise SimulatorOutputError("experiment.html must inline the simulation script.")
        if not self._SETUP_PATTERN.search(inline_script):
            raise SimulatorOutputError("experiment.html script must define p5 setup().")
        if not self._DRAW_PATTERN.search(inline_script):
            raise SimulatorOutputError("experiment.html script must define p5 draw().")
        if "createCanvas" not in inline_script and "createcanvas" not in inline_script.lower():
            raise SimulatorOutputError("experiment.html script must create a p5 canvas.")

    @staticmethod
    def default_index_html(title: str = "Virtual Lab Simulation") -> str:
        return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/p5@1.9.4/lib/p5.min.js"></script>
  <script src="sketch.js"></script>
  <style>
    html,
    body {{
      margin: 0;
      min-height: 100%;
      background: #111;
    }}

    body {{
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: auto;
    }}

    canvas {{
      display: block;
      max-width: 100vw;
      max-height: 100vh;
    }}
  </style>
</head>
<body>
</body>
</html>
"""

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return json.dumps(value, ensure_ascii=False, indent=2)

    @staticmethod
    def _load_json_payload(raw: str) -> dict[str, Any] | None:
        """Decode the first balanced JSON object in raw output."""
        text = raw.strip()
        if not text:
            return None

        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    @staticmethod
    def _strip_outer_fence(text: str) -> str:
        text = (text or "").strip()
        match = re.fullmatch(r"```[a-zA-Z0-9_-]*\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    @staticmethod
    def _extract_named_block(raw: str, filename: str) -> str:
        pattern = re.compile(
            rf"{re.escape(filename)}\s*:?\s*```[a-zA-Z0-9_-]*\s*\n(.*?)```",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(raw)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_inline_script(index_html: str) -> str:
        scripts = re.findall(
            r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
            index_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for script in scripts:
            if "function setup" in script or "createCanvas" in script:
                return script.strip()
        return ""

    @classmethod
    def _ensure_p5_index(cls, index_html: str) -> str:
        html = index_html.strip()
        if not html:
            return cls.default_index_html()

        if "p5" not in html.lower():
            html = cls._insert_before(
                html,
                "</head>",
                '  <script src="https://cdn.jsdelivr.net/npm/p5@1.9.4/lib/p5.min.js"></script>\n',
            )

        if "sketch.js" not in html:
            html = cls._insert_before(
                html,
                "</head>",
                '  <script src="sketch.js"></script>\n',
            )

        return html

    @classmethod
    def _ensure_p5_html(cls, experiment_html: str) -> str:
        html = experiment_html.strip()
        if not html:
            return html

        if "p5" not in html.lower():
            html = cls._insert_before(
                html,
                "</head>",
                '  <script src="https://cdn.jsdelivr.net/npm/p5@1.9.4/lib/p5.min.js"></script>\n',
            )
        return html

    @staticmethod
    def _insert_before(html: str, marker: str, snippet: str) -> str:
        idx = html.lower().find(marker.lower())
        if idx >= 0:
            return html[:idx] + snippet + html[idx:]
        return snippet + html


class SimulatorAgent(BaseAgent):
    """
    Agent that receives a grounded experiment script and returns runnable HTML.
    """

    _SYSTEM_PROMPT = (
        "You are simulator_agent in a multi-agent Virtual Science Lab system. "
        "You are a senior p5.js simulation engineer and science-education "
        "simulation designer.\n\n"
        "Your job is to convert a grounded experiment script plus retrieved RAG "
        "context into one executable simulator HTML file.\n\n"
        "GROUNDING RULES:\n"
        "1) Treat Retrieved Scientific Context as the authoritative source. The "
        "scripting output is useful, but RAG wins if there is any conflict.\n"
        "2) Extract every scientific fact, numeric value, unit, relationship, "
        "constraint, safety rule, and expected observation from the RAG context.\n"
        "3) Every extracted fact must be represented in code constants, visible "
        "labels/HUD/table, interaction behavior, or simulation logic. Do not leave "
        "facts only in prose.\n"
        "4) If a necessary simulator-only value is missing from RAG, choose a "
        "reasonable standard simulation value and label it exactly as "
        "[Standard Simulation Values] in the structured understanding and code "
        "comments. Never invent educational facts.\n"
        "5) Preserve source trace labels in a compact on-screen or code-level "
        "grounding map, so reviewers can connect claims to source chunks/pages.\n\n"
        "SIMULATOR RULES:\n"
        "- Use p5.js global mode with function setup() and function draw().\n"
        "- The simulation must be dynamic, interactive, and visually explanatory.\n"
        "- Use deterministic, named state variables for the core science model.\n"
        "- Prefer data-driven rendering from extracted RAG constants.\n"
        "- Include teacher/student friendly Vietnamese UI text where visible.\n"
        "- Return one self-contained HTML document with all JavaScript inlined. "
        "The only external dependency allowed is the p5.js CDN script.\n\n"
        "OUTPUT CONTRACT:\n"
        "Return ONLY one valid JSON object. Do not wrap it in Markdown fences. "
        "The JSON object must follow this schema:\n"
        "{\n"
        '  "structured_understanding": "exhaustive extracted facts and simulation intent",\n'
        '  "dsl": "entities, parameters, physics/state rules, interactions, visualization",\n'
        '  "architecture": "module-level design of the generated inline HTML simulation",\n'
        '  "files": {\n'
        '    "experiment.html": "<full standalone HTML document loading p5.js CDN and inlining all simulation JavaScript>"\n'
        "  },\n"
        '  "validation_checklist": ["fact coverage checks", "runtime checks"]\n'
        "}"
    )

    def __init__(
        self,
        llm_model: Optional[str] = None,
        *,
        client: Optional[_ChatClient] = None,
        output_parser: Optional[SimulatorOutputParser] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.client: Optional[_ChatClient] = client
        self.output_parser = output_parser or SimulatorOutputParser()

        if self.client is not None:
            return

        try:
            resolved_max_tokens = max_tokens or int(
                os.getenv("SIMULATOR_MAX_TOKENS", "12000")
            )
            resolved_model = (
                llm_model
                or os.getenv("SIMULATOR_LLM_MODEL")
                or os.getenv("LLM_MODEL")
                or ""
            )
            provider = get_provider(
                backend=os.getenv("SIMULATOR_PROVIDER_BACKEND"),
                model=resolved_model or None,
                temperature=self._compatible_temperature(resolved_model, temperature),
                max_tokens=resolved_max_tokens,
                timeout=float(os.getenv("SIMULATOR_TIMEOUT", "300")),
            )
            self.client = OpenAIClient(provider)
        except Exception as exc:
            logger.warning("SimulatorAgent: failed to initialize LLM client: %s", exc)
            self.client = None

    @staticmethod
    def _compatible_temperature(model_name: str, requested: float) -> float:
        """Use API default temperature for models that reject custom sampling."""
        markers = ("gpt-5", "o1", "o3", "o4")
        lowered = model_name.lower()
        if any(marker in lowered for marker in markers):
            return 1.0
        return requested

    def run(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
        messages: Optional[list[BaseMessage]] = None,
    ) -> str:
        """Return raw simulator LLM output for BaseAgent compatibility."""
        if not self.client:
            return "SimulatorAgent is not properly configured (client missing)."

        prompt = self._build_user_prompt(query, context or {})
        response = self.client.chat(
            [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        return str(getattr(response, "content", response)).strip()

    def generate(
        self,
        *,
        teacher_prompt: str,
        experiment_script: str,
        retrieved_context: str = "",
        retrieved_sources: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> SimulatorArtifacts:
        """Generate and parse simulator artifacts."""
        if not experiment_script.strip():
            raise SimulatorOutputError("SimulatorAgent requires an experiment script.")

        payload = {
            **(context or {}),
            "experiment_script": experiment_script,
            "retrieved_context": retrieved_context,
            "retrieved_sources": retrieved_sources,
        }
        raw = self.run(teacher_prompt, context=payload)
        if raw.startswith("SimulatorAgent is not properly configured"):
            raise SimulatorOutputError(raw)
        return self.output_parser.parse(raw)

    @staticmethod
    def _build_user_prompt(query: str, context: dict[str, Any]) -> str:
        experiment_script = str(context.get("experiment_script", "")).strip()
        retrieved_context = str(context.get("retrieved_context", "")).strip()
        retrieved_sources = str(context.get("retrieved_sources", "")).strip()

        extra_context = {
            key: value
            for key, value in context.items()
            if key
            not in {
                "experiment_script",
                "retrieved_context",
                "retrieved_sources",
            }
        }

        return "\n\n".join(
            [
                f"Teacher Prompt:\n{query.strip()}",
                f"Scripting Agent Output:\n{experiment_script or '(missing)'}",
                (
                    "Retrieved Scientific Context:\n"
                    f"{retrieved_context or '(not supplied; rely on script only)'}"
                ),
                f"Retrieved Source Trace:\n{retrieved_sources or '(not supplied)'}",
                (
                    "Additional Orchestration Context:\n"
                    f"{json.dumps(extra_context, ensure_ascii=False, indent=2)}"
                ),
                (
                    "Generate the JSON output contract now. The files must run "
                    "as one standalone experiment.html file in a browser."
                ),
            ]
        )
