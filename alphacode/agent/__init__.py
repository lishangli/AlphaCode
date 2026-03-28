"""
Agent module for ALPHACODE.

Provides different agent types:
- BaseAgent: Base class with session management
- ConversationAgent: For questions and chitchat
- CodeAgent: For code generation
"""

from alphacode.agent.base import AgentResponse, BaseAgent
from alphacode.agent.code import CodeAgent
from alphacode.agent.conversation import ConversationAgent
from alphacode.agent.prompts import PromptTemplates

__all__ = [
    "AgentResponse",
    "BaseAgent",
    "ConversationAgent",
    "CodeAgent",
    "PromptTemplates",
]
