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

logger = logging.getLogger(__name__)


class ScriptingAgent(BaseAgent):
    """
    Agent responsible for writing and formatting step-by-step educational guides,
    safety procedures, and theoretical background for science experiments.
    """

    _SYSTEM_PROMPT = (
        "You are an expert science educator and lab instructor. Write a comprehensive, step-by-step "
        "experimental guide for the scientific topic requested by the user. "
        "Include the objective, materials needed, safety precautions, theoretical background, "
        "and detailed step-by-step procedures. Format your guide clearly in Markdown."
    )

    def __init__(self, llm_model: Optional[str] = None) -> None:
        self.chain = None
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
                    ("human", "Write an experiment guide for the following request:\n\n{query}\n\nAdditional Context: {context_notes}"),
                ]
            )
            self.chain = prompt | llm | StrOutputParser()
        except Exception as exc:
            logger.warning(f"ScriptingAgent: failed to initialize chain: {exc}")

    def run(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
        messages: Optional[list[BaseMessage]] = None,
    ) -> str:
        if not self.chain:
            return "ScriptingAgent is not properly configured (chain missing)."

        context = context or {}
        context_notes = "\n".join(f"{k}: {v}" for k, v in context.items()) if context else "None"

        try:
            return self.chain.invoke({"query": query, "context_notes": context_notes})
        except Exception as exc:
            logger.error(f"ScriptingAgent error: {exc}")
            return f"Guide generation failed: {exc}"
