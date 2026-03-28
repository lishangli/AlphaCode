"""
Unified Agent for ALPHACODE.

This is a compatibility layer that wraps the new agent modules.
New code should use alphacode.agent directly.
"""

import logging

from alphacode.agent.base import AgentResponse
from alphacode.agent.code import CodeAgent
from alphacode.agent.conversation import ConversationAgent
from alphacode.llm.client import LLMClient
from alphacode.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)

# Re-export for compatibility
__all__ = ["Agent", "AgentResponse", "ConversationAgent", "CodeAgent"]


class Agent(ConversationAgent):
    """
    Unified agent that handles all intents with tool support.

    This is a compatibility wrapper around ConversationAgent.
    New code should use ConversationAgent or CodeAgent directly.
    """

    # System prompts (for backward compatibility)
    CONVERSATION_SYSTEM = ConversationAgent.SYSTEM_PROMPT
    CODE_SYSTEM = CodeAgent.SYSTEM_PROMPT

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
        max_tool_iterations: int = 5,
    ):
        """
        Initialize agent.

        Args:
            llm_client: LLM client for generation
            tool_executor: Tool executor for running tool calls
            max_tool_iterations: Maximum tool call iterations per request
        """
        super().__init__(
            llm_client=llm_client,
            tool_executor=tool_executor,
            session_manager=None,  # Will be created if needed
            max_tool_iterations=max_tool_iterations,
        )
