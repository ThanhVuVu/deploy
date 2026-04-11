"""
tests/test_llm_output_parser.py

Unit tests for src/llm/output_parser.py

Run with:
    pytest tests/test_llm_output_parser.py -v
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.llm.output_parser import OutputParser, ParsedOutput


# ---------------------------------------------------------------------------
# ParsedOutput helpers
# ---------------------------------------------------------------------------

class TestParsedOutputFirstBlock:
    def test_first_block_no_filter(self):
        raw = "```python\nprint('hi')\n```"
        result = OutputParser.parse(raw)
        assert result.first_block() == "print('hi')"

    def test_first_block_with_matching_lang(self):
        raw = "```cypher\nMATCH (n) RETURN n\n```"
        result = OutputParser.parse(raw)
        assert result.first_block("cypher") == "MATCH (n) RETURN n"

    def test_first_block_wrong_lang_returns_none(self):
        raw = "```python\nprint('hi')\n```"
        result = OutputParser.parse(raw)
        assert result.first_block("cypher") is None

    def test_first_block_empty_when_no_blocks(self):
        result = OutputParser.parse("just plain text")
        assert result.first_block() is None

    def test_first_block_lang_case_insensitive(self):
        raw = "```Python\ncode\n```"
        result = OutputParser.parse(raw)
        assert result.first_block("python") == "code"


class TestParsedOutputAsModel:
    class _Schema(BaseModel):
        action: str
        value: int

    def test_as_model_success(self):
        raw = '```json\n{"action": "run", "value": 42}\n```'
        result = OutputParser.parse(raw)
        obj = result.as_model(self._Schema)
        assert obj.action == "run"
        assert obj.value == 42

    def test_as_model_no_json_raises(self):
        result = OutputParser.parse("no json here")
        with pytest.raises(ValueError, match="No JSON data found"):
            result.as_model(self._Schema)

    def test_as_model_schema_mismatch_raises(self):
        raw = '```json\n{"wrong_field": true}\n```'
        result = OutputParser.parse(raw)
        with pytest.raises(ValueError, match="did not match schema"):
            result.as_model(self._Schema)


class TestParsedOutputRepr:
    def test_repr_contains_info(self):
        result = OutputParser.parse("```json\n{}\n```")
        r = repr(result)
        assert "ParsedOutput" in r
        assert "has_json=True" in r


# ---------------------------------------------------------------------------
# OutputParser.parse — code block extraction
# ---------------------------------------------------------------------------

class TestOutputParserCodeBlocks:
    def test_single_python_block(self):
        raw = "```python\nx = 1\n```"
        result = OutputParser.parse(raw)
        assert len(result.code_blocks) == 1
        assert result.code_blocks[0]["lang"] == "python"
        assert "x = 1" in result.code_blocks[0]["body"]

    def test_multiple_blocks(self):
        raw = "```python\nx = 1\n```\nsome text\n```json\n{}\n```"
        result = OutputParser.parse(raw)
        assert len(result.code_blocks) == 2

    def test_block_with_no_lang_tag(self):
        raw = "```\nraw content\n```"
        result = OutputParser.parse(raw)
        assert len(result.code_blocks) == 1
        assert result.code_blocks[0]["lang"] == "text"

    def test_no_blocks(self):
        result = OutputParser.parse("Just plain text.")
        assert result.code_blocks == []


# ---------------------------------------------------------------------------
# OutputParser.parse — fence stripping
# ---------------------------------------------------------------------------

class TestOutputParserStripFences:
    def test_fence_removed_from_text(self):
        raw = "Prefix\n```python\ncode\n```\nSuffix"
        result = OutputParser.parse(raw)
        assert "```" not in result.text
        assert "Prefix" in result.text
        assert "Suffix" in result.text

    def test_blank_text_when_only_fences(self):
        raw = "```python\ncode\n```"
        result = OutputParser.parse(raw)
        # code_blocks should be extracted; text should be empty/minimal
        assert "code" not in result.text

    def test_excessive_blank_lines_collapsed(self):
        raw = "Line 1\n\n\n\n\nLine 2"
        result = OutputParser.parse(raw)
        # Should not have 3+ consecutive newlines
        assert "\n\n\n" not in result.text


# ---------------------------------------------------------------------------
# OutputParser.parse — JSON extraction
# ---------------------------------------------------------------------------

class TestOutputParserJsonExtraction:
    def test_json_from_json_tagged_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        result = OutputParser.parse(raw)
        assert result.json == {"key": "value"}

    def test_json_from_non_json_fence(self):
        raw = '```text\n{"key": "fallback"}\n```'
        result = OutputParser.parse(raw)
        assert result.json == {"key": "fallback"}

    def test_json_from_bare_text(self):
        raw = 'The answer is: {"score": 99}'
        result = OutputParser.parse(raw)
        assert result.json == {"score": 99}

    def test_json_list(self):
        raw = '```json\n[1, 2, 3]\n```'
        result = OutputParser.parse(raw)
        assert result.json == [1, 2, 3]

    def test_no_json_returns_none(self):
        result = OutputParser.parse("There is no JSON here.")
        assert result.json is None

    def test_malformed_json_returns_none(self):
        raw = '```json\n{not: valid json}\n```'
        result = OutputParser.parse(raw)
        assert result.json is None

    def test_json_tagged_fence_takes_priority_over_bare(self):
        raw = '{"bare": true}\n```json\n{"fenced": true}\n```'
        result = OutputParser.parse(raw)
        assert result.json == {"fenced": True}

    def test_nested_json(self):
        raw = '```json\n{"a": {"b": [1, 2]}}\n```'
        result = OutputParser.parse(raw)
        assert result.json == {"a": {"b": [1, 2]}}


# ---------------------------------------------------------------------------
# OutputParser class-level shortcuts
# ---------------------------------------------------------------------------

class TestOutputParserShortcuts:
    def test_extract_code_by_lang(self):
        raw = "```cypher\nMATCH (n) RETURN n\n```"
        code = OutputParser.extract_code(raw, lang="cypher")
        assert code == "MATCH (n) RETURN n"

    def test_extract_code_no_match_returns_none(self):
        raw = "```python\ncode\n```"
        assert OutputParser.extract_code(raw, lang="sql") is None

    def test_clean_removes_fences(self):
        raw = "Before\n```python\ncode\n```\nAfter"
        clean = OutputParser.clean(raw)
        assert "```" not in clean
        assert "Before" in clean
        assert "After" in clean

    def test_parse_as_success(self):
        class MyModel(BaseModel):
            name: str
            count: int

        raw = '```json\n{"name": "test", "count": 7}\n```'
        obj = OutputParser.parse_as(raw, MyModel)
        assert obj.name == "test"
        assert obj.count == 7

    def test_parse_as_no_json_raises(self):
        class MyModel(BaseModel):
            name: str

        with pytest.raises(ValueError):
            OutputParser.parse_as("no json", MyModel)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self):
        result = OutputParser.parse("")
        assert result.text == ""
        assert result.code_blocks == []
        assert result.json is None

    def test_only_whitespace(self):
        result = OutputParser.parse("   \n\n  ")
        assert result.text == ""

    def test_real_llm_style_reply(self):
        raw = """
Sure! Here is the Cypher query:

```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE m.released > 2000
RETURN p.name, m.title
```

Let me know if you need adjustments.
        """
        result = OutputParser.parse(raw)
        assert result.first_block("cypher") is not None
        assert "MATCH" in result.first_block("cypher")
        assert "```" not in result.text
        assert "Let me know" in result.text

    def test_multiple_json_blocks_first_wins(self):
        raw = '```json\n{"first": 1}\n```\n```json\n{"second": 2}\n```'
        result = OutputParser.parse(raw)
        assert result.json == {"first": 1}
