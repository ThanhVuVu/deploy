"""
src/agents/scripting/scripting_agent.py

Scripting agent responsible for writing comprehensive guides and step-by-step procedures for laboratory experiments.
"""

import logging
from typing import Any, Optional

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.llm.provider import get_provider
from src.agents.base.base_agent import BaseAgent
from src.agents.scripting.rag import ScriptingScienceRetriever

logger = logging.getLogger(__name__)


class ScriptingAgent(BaseAgent):
    """
    Agent responsible for writing and formatting step-by-step educational guides,
    safety procedures, and theoretical background for science experiments.
    """

    _NO_DATA_MESSAGE = "I don't have enough scientific data"
    _SYSTEM_PROMPT = (
        "You are scripting_agent in a multi-agent virtual lab system for middle school science. "
        "Your output is consumed by simulator_agent.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1) Use ONLY facts present in the section 'Retrieved Scientific Context'.\n"
        "2) Never add external knowledge, invented numbers, or assumptions.\n"
        "3) Every formula, threshold, and numeric parameter must be traceable to retrieved context.\n"
        "4) If context is missing or insufficient for the teacher request, output exactly: "
        "I don't have enough scientific data\n\n"
        "OUTPUT REQUIREMENTS (Markdown):\n"
        "- Experiment objective\n"
        "- Materials and setup with exact values\n"
        "- Simulator instructions step-by-step\n"
        "- Expected observable phenomena (visual/audio/data cues)\n"
        "- Safety constraints\n"
        "- Grounding notes mapping key claims to source chunks"
    )

    def __init__(
        self,
        llm_model: Optional[str] = None,
        retriever: Optional[ScriptingScienceRetriever] = None,
    ) -> None:
        self.chain = None
        self.retriever = None

        try:
            provider = get_provider(model=llm_model)
            llm = ChatOpenAI(
                model=provider.model,
                api_key=provider.api_key,
                base_url=provider.base_url,
                temperature=0.2,
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self._SYSTEM_PROMPT),
                    (
                        "human",
                        "Teacher Request:\n{query}\n\n"
                        "Retrieved Scientific Context (authoritative source):\n{retrieved_context}\n\n"
                        "Retrieved Source Trace:\n{retrieved_sources}\n\n"
                        "Additional Orchestration Context:\n{context_notes}\n\n"
                        "If context is insufficient, output exactly: I don't have enough scientific data",
                    ),
                ]
            )
            self.chain = prompt | llm | StrOutputParser()
        except Exception as exc:
            logger.warning(f"ScriptingAgent: failed to initialize chain: {exc}")

        try:
            self.retriever = retriever or ScriptingScienceRetriever()
        except Exception as exc:
            logger.warning(f"ScriptingAgent: failed to initialize retriever: {exc}")

    def run(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
        messages: Optional[list[BaseMessage]] = None,
    ) -> str:
        if not self.chain:
            return "ScriptingAgent is not properly configured (chain missing)."

        if not self.retriever:
            return self._NO_DATA_MESSAGE

        context = context or {}
        context_notes = "\n".join(f"{k}: {v}" for k, v in context.items()) if context else "None"
        top_k = self._resolve_top_k(context)

        try:
            retrieval = self.retriever.retrieve(query, top_k=top_k)
            if not retrieval.has_enough_data:
                return self._NO_DATA_MESSAGE

            return self.chain.invoke(
                {
                    "query": query,
                    "context_notes": context_notes,
                    "retrieved_context": retrieval.grounded_context,
                    "retrieved_sources": retrieval.source_report,
                }
            )
        except Exception as exc:
            logger.error(f"ScriptingAgent error: {exc}")
            return f"Guide generation failed: {exc}"

    @staticmethod
    def _resolve_top_k(context: dict[str, Any]) -> int:
        raw_value = context.get("rag_top_k", 4)
        try:
            top_k = int(raw_value)
        except (TypeError, ValueError):
            return 4
        return max(1, min(top_k, 8))
