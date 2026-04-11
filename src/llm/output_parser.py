"""
src/llm/output_parser.py

Structured post-processing of raw LLM text.

The parser can:
  - Strip code fences (```python … ```, ```cypher … ```, etc.)
  - Extract JSON blobs
  - Validate against a Pydantic model
  - Split multi-block responses

Usage
-----
    from src.llm.output_parser import OutputParser, ParsedOutput

    raw = '''
    Here is the answer:
    ```json
    {"action": "run", "steps": 3}
    ```
    '''

    result: ParsedOutput = OutputParser.parse(raw)
    print(result.text)  # clean text without fences
    print(result.json)  # {"action": "run", "steps": 3}

    # Or validate against a Pydantic schema:
    from pydantic import BaseModel

    class MySchema(BaseModel):
        action: str
        steps: int

    obj: MySchema = OutputParser.parse_as(raw, MySchema)
    print(obj.action)   # "run"
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches ``` lang\n ... ``` or ``` ... ```
_FENCE_PATTERN = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_\-]*)\s*\n?(?P<body>.*?)```",
    re.DOTALL,
)

# Matches a bare JSON object or array (greedy, first occurrence)
_JSON_PATTERN = re.compile(
    r"(?P<json>\{.*?\}|\[.*?\])",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# ParsedOutput
# ---------------------------------------------------------------------------

class ParsedOutput:
    """
    The result of running :class:`OutputParser` over LLM output text.

    Attributes
    ----------
    raw : str
        The original, unmodified LLM output.
    text : str
        The response with all code fences stripped and whitespace normalised.
    code_blocks : list[dict]
        Every fenced block found, as ``{"lang": "python", "body": "…"}``.
    json : dict | list | None
        The first valid JSON object/array found anywhere in the response
        (inside or outside a fence), or ``None`` if none is found.
    """

    def __init__(
        self,
        raw: str,
        text: str,
        code_blocks: list[dict[str, str]],
        json_data: Optional[Any],
    ) -> None:
        self.raw = raw
        self.text = text
        self.code_blocks = code_blocks
        self.json = json_data

    # ------------------------------------------------------------------ #
    # Convenience                                                           #
    # ------------------------------------------------------------------ #

    def first_block(self, lang: Optional[str] = None) -> Optional[str]:
        """
        Return the body of the first code block, optionally filtered by language.

        Parameters
        ----------
        lang:
            Case-insensitive language tag (e.g. ``"python"``, ``"cypher"``).
            Pass ``None`` to return the first block regardless of language.
        """
        for block in self.code_blocks:
            if lang is None or block["lang"].lower() == lang.lower():
                return block["body"].strip()
        return None

    def as_model(self, model_cls: Type[T]) -> T:
        """
        Deserialise :attr:`json` into *model_cls*.

        Raises
        ------
        ValueError
            If :attr:`json` is ``None`` or fails Pydantic validation.
        """
        if self.json is None:
            raise ValueError("No JSON data found in the LLM output.")
        try:
            return model_cls.model_validate(self.json)
        except ValidationError as exc:
            raise ValueError(
                f"LLM output did not match schema {model_cls.__name__}: {exc}"
            ) from exc

    def __str__(self) -> str:  # noqa: D105
        return self.text

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ParsedOutput("
            f"text_len={len(self.text)}, "
            f"code_blocks={len(self.code_blocks)}, "
            f"has_json={self.json is not None})"
        )


# ---------------------------------------------------------------------------
# OutputParser
# ---------------------------------------------------------------------------

class OutputParser:
    """
    Static utility class for parsing and cleaning LLM text output.

    All methods are class-level — no instantiation needed.
    """

    # ------------------------------------------------------------------ #
    # Main entry points                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def parse(cls, text: str) -> ParsedOutput:
        """
        Parse *text* and return a :class:`ParsedOutput`.

        Steps:
          1. Extract all fenced code blocks.
          2. Create a clean ``text`` version by removing fences.
          3. Attempt JSON extraction from code blocks first, then bare text.
        """
        code_blocks = cls._extract_code_blocks(text)
        clean_text = cls._strip_fences(text)
        json_data = cls._extract_json(text, code_blocks)

        return ParsedOutput(
            raw=text,
            text=clean_text,
            code_blocks=code_blocks,
            json_data=json_data,
        )

    @classmethod
    def parse_as(cls, text: str, model_cls: Type[T]) -> T:
        """
        Parse *text* and immediately validate the JSON payload as *model_cls*.

        Parameters
        ----------
        text:
            Raw LLM output string.
        model_cls:
            A :class:`pydantic.BaseModel` subclass to validate against.

        Raises
        ------
        ValueError
            If no JSON is found or validation fails.
        """
        parsed = cls.parse(text)
        return parsed.as_model(model_cls)

    @classmethod
    def extract_code(cls, text: str, lang: Optional[str] = None) -> Optional[str]:
        """
        Shortcut: return the first code block body (optionally by language).

        Returns ``None`` if no matching block exists.
        """
        return cls.parse(text).first_block(lang)

    @classmethod
    def clean(cls, text: str) -> str:
        """Return *text* with code fences stripped and whitespace normalised."""
        return cls._strip_fences(text)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def _extract_code_blocks(cls, text: str) -> list[dict[str, str]]:
        """Return every fenced block as ``[{"lang": str, "body": str}, …]``."""
        blocks = []
        for match in _FENCE_PATTERN.finditer(text):
            blocks.append(
                {
                    "lang": match.group("lang") or "text",
                    "body": match.group("body"),
                }
            )
        return blocks

    @classmethod
    def _strip_fences(cls, text: str) -> str:
        """Remove all fenced code blocks from *text* and normalise whitespace."""
        stripped = _FENCE_PATTERN.sub("", text)
        # Collapse excessive blank lines
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        return stripped.strip()

    @classmethod
    def _extract_json(
        cls,
        text: str,
        code_blocks: list[dict[str, str]],
    ) -> Optional[Any]:
        """
        Try to deserialise JSON from:
          1. Blocks tagged ``json`` or ``JSON``
          2. Any other code block body
          3. Bare JSON patterns in the full text
        Returns the first successfully parsed value, or ``None``.
        """
        # Priority 1: explicit json-tagged fences
        for block in code_blocks:
            if block["lang"].lower() == "json":
                parsed = cls._try_parse_json(block["body"])
                if parsed is not None:
                    return parsed

        # Priority 2: all other fences
        for block in code_blocks:
            if block["lang"].lower() != "json":
                parsed = cls._try_parse_json(block["body"])
                if parsed is not None:
                    return parsed

        # Priority 3: bare JSON in the raw text
        for match in _JSON_PATTERN.finditer(text):
            parsed = cls._try_parse_json(match.group("json"))
            if parsed is not None:
                return parsed

        return None

    @staticmethod
    def _try_parse_json(raw: str) -> Optional[Any]:
        """Attempt ``json.loads``, returning ``None`` on failure."""
        raw = raw.strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Failed to parse JSON snippet: %r", raw[:80])
            return None
