"""
Agent base module for ALPHACODE.

Provides the base agent class with session management support.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from alphacode.llm.client import LLMClient, LLMResponse
from alphacode.state.session_manager import SessionManager, SessionState
from alphacode.tools.executor import TOOL_DEFINITIONS, ToolExecutor

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Agent response with tool execution results."""
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    session_id: str = ""


class BaseAgent(ABC):
    """
    Base agent class with session management.

    All agents share:
    - LLM client
    - Tool executor
    - Session manager
    """

    SYSTEM_PROMPT = "You are ALPHACODE, a helpful AI assistant."

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
        session_manager: SessionManager = None,
        max_tool_iterations: int = 5,
    ):
        """
        Initialize base agent.

        Args:
            llm_client: LLM client for generation
            tool_executor: Tool executor for running tool calls
            session_manager: Optional session manager
            max_tool_iterations: Maximum tool call iterations
        """
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.session_manager = session_manager
        self.max_tool_iterations = max_tool_iterations
        self.current_session: SessionState | None = None

    def start_session(self, intent: str = "conversation", goal: str = "") -> SessionState:
        """Start a new session."""
        if self.session_manager:
            self.current_session = self.session_manager.create_session(
                intent=intent,
                goal=goal,
            )
        return self.current_session

    def record_message(self, role: str, content: str, metadata: dict = None):
        """Record a message in the session."""
        if self.session_manager and self.current_session:
            self.session_manager.record_message(role, content, metadata)

    def record_tool_call(
        self,
        tool_name: str,
        args: dict,
        result: str,
        success: bool = True,
    ):
        """Record a tool call in the session."""
        if self.session_manager and self.current_session:
            self.session_manager.record_tool_call(
                tool_name, args, result, success
            )

    @abstractmethod
    async def process(self, user_input: str, **kwargs) -> AgentResponse:
        """Process user input. Must be implemented by subclasses."""
        pass

    def process_sync(self, user_input: str, **kwargs) -> AgentResponse:
        """Synchronous version of process."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.process(user_input, **kwargs)
        )

    async def _call_llm_with_tools(
        self,
        user_input: str,
        system_prompt: str = None,
        tools: list[dict] = None,
        messages: list[dict] = None,
    ) -> LLMResponse:
        """
        Call LLM with tool support.

        Args:
            user_input: User input (used if messages is None)
            system_prompt: System prompt
            tools: Tool definitions
            messages: Pre-built messages (overrides user_input)

        Returns:
            LLMResponse
        """
        system = system_prompt or self.SYSTEM_PROMPT
        available_tools = tools or TOOL_DEFINITIONS

        # Build messages
        if messages is None:
            messages = [{"role": "user", "content": user_input}]

        full_messages = [{"role": "system", "content": system}]
        full_messages.extend(messages)

        # Use OpenAI client
        client = self.llm_client._get_client()
        start = time.time()

        response = await client.chat.completions.create(
            model=self.llm_client.model,
            messages=full_messages,
            tools=available_tools,
            tool_choice="auto",
            temperature=self.llm_client.temperature,
            max_tokens=self.llm_client.max_tokens,
        )

        choice = response.choices[0]
        message = choice.message

        # Extract content
        content = message.content or ""

        # Extract tool calls
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "tool": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                })

        return LLMResponse(
            content=content,
            model=response.model or self.llm_client.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            latency=time.time() - start,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
        )

    async def _execute_tool_call(
        self,
        tool_name: str,
        args: dict,
    ) -> tuple[bool, str]:
        """
        Execute a tool call.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Tuple of (success, result_string)
        """
        result = self.tool_executor.execute({
            "tool": tool_name,
            "args": args,
        })

        return result.success, result.output if result.success else f"Error: {result.error}"
