from .science_retriever import RetrievalResult, RetrievedChunk, ScriptingScienceRetriever
from .multimodal_indexer import MultimodalIndexer, IngestStats
from .multimodal_retriever import (
    MultimodalRetriever,
    MultimodalRetrievalResult,
    TextChunk,
    ImageChunk,
    TableChunk,
)

__all__ = [
    # Legacy text-only retriever (backward compat)
    "RetrievalResult",
    "RetrievedChunk",
    "ScriptingScienceRetriever",
    # Multimodal stack
    "MultimodalIndexer",
    "IngestStats",
    "MultimodalRetriever",
    "MultimodalRetrievalResult",
    "TextChunk",
    "ImageChunk",
    "TableChunk",
]
