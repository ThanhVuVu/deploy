"""
src/llm/__init__.py
LLM package — exposes the provider registry, typed client, and output parser.
"""

from .provider import LLMProvider, get_provider
from .openai_client import OpenAIClient
from .output_parser import OutputParser, ParsedOutput

__all__ = [
    "LLMProvider",
    "get_provider",
    "OpenAIClient",
    "OutputParser",
    "ParsedOutput",
]
