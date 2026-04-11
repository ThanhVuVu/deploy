"""
src/agents/base/base_agent.py

Abstract base class for all domain-specific agents in the simulator.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.messages import BaseMessage


class BaseAgent(ABC):
    """
    Abstract interface for a science lab agent.
    """

    @abstractmethod
    def run(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
        messages: Optional[list[BaseMessage]] = None,
    ) -> str:
        """
        Execute the agent logic for a given query.

        Parameters
        ----------
        query: str
            The user's input/question.
        context: dict
            Additional state or variables from the orchestration graph.
        messages: list[BaseMessage]
            Pre-existing conversation history.

        Returns
        -------
        str
            The agent's text response.
        """
        pass
