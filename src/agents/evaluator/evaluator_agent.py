"""
src/agents/evaluator/evaluator_agent.py

Evaluator agent used to assess student answers or experiment outcomes.
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


class EvaluatorAgent(BaseAgent):
    """
    Agent responsible for evaluating responses or experiment setups from a correctness and safety perspective.
    """

    _SYSTEM_PROMPT = (
        "You are an expert science evaluator. Your job is to review the experiment setup or answer "
        "provided by the user and evaluate it for scientific accuracy, logic, and safety. "
        "Provide a concise summary of your assessment, pointing out any mistakes or hazards."
    )

    def __init__(self, llm_model: Optional[str] = None) -> None:
        self.chain = None
        try:
            provider = get_provider(model=llm_model)
            llm = ChatOpenAI(
                model=provider.model,
                api_key=provider.api_key,
                base_url=provider.base_url,
                temperature=0.1,  # low temperature for stable evaluation
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self._SYSTEM_PROMPT),
                    ("human", "Evaluate the following:\n\n{query}\n\nContext:\n{context_notes}"),
                ]
            )
            self.chain = prompt | llm | StrOutputParser()
        except Exception as exc:
            logger.warning(f"EvaluatorAgent: failed to initialize chain: {exc}")

    def run(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
        messages: Optional[list[BaseMessage]] = None,
    ) -> str:
        if not self.chain:
            return "EvaluatorAgent is not properly configured (chain missing)."

        context = context or {}
        context_notes = "\n".join(f"{k}: {v}" for k, v in context.items()) if context else "None"

        try:
            return self.chain.invoke({"query": query, "context_notes": context_notes})
        except Exception as exc:
            logger.error(f"EvaluatorAgent error: {exc}")
            return f"Evaluation failed: {exc}"
