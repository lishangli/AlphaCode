"""
Unified Agent for ALPHACODE.

Handles all intents with tool support. LLM decides which tools to use.
MCTS exploration is only used for code generation tasks.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from alphacode.llm.client import LLMClient, LLMResponse
from alphacode.tools.executor import TOOL_DEFINITIONS, ToolExecutor

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Agent response with tool execution results."""
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0


class Agent:
    """
    Unified agent that handles all intents with tool support.

    The LLM decides which tools to use based on the user's request.
    This works for:
    - Questions: LLM may use read/grep/glob to find information
    - Chitchat: LLM likely uses no tools
    - Code tasks: LLM uses write/edit/bash, then MCTS takes over for optimization
    """

    # System prompts for different modes
    CONVERSATION_SYSTEM = """You are ALPHACODE, a helpful AI assistant with access to tools.

You can help with:
- Answering programming questions
- Exploring and explaining code
- Writing and modifying code
- Running tests and commands

You have access to tools for:
- Reading and writing files
- Searching code
- Running shell commands

Use tools when they would help answer the user's question or complete a task.
For simple questions or chat, you can respond directly without tools.

IMPORTANT: After using tools, ALWAYS show the results to the user!
- If you read a file, show the file contents
- If you search for files, list the files found
- If you run a command, show the output

Be thorough but concise. Format code and file contents nicely."""

    CODE_SYSTEM = """You are ALPHACODE, a code generation assistant.

You have access to tools for creating and modifying code.
Use the appropriate tools to implement the requested functionality.

After writing code, consider:
- Does it handle edge cases?
- Is it readable and maintainable?
- Should there be tests?

Create working, production-quality code."""

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
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_tool_iterations = max_tool_iterations
        self.conversation_history: list[dict] = []

    async def process(
        self,
        user_input: str,
        system_prompt: str = None,
        tools: list[dict] = None,
    ) -> AgentResponse:
        """
        Process user input with tool calling loop.

        The LLM can call tools, see results, and decide to call more tools
        or provide a final response.

        Args:
            user_input: User's message
            system_prompt: Optional custom system prompt
            tools: Optional custom tool definitions

        Returns:
            AgentResponse with final content and tool execution details
        """
        system = system_prompt or self.CONVERSATION_SYSTEM
        available_tools = tools or TOOL_DEFINITIONS

        # Build initial messages
        messages = [{"role": "user", "content": user_input}]

        all_tool_calls = []
        all_tool_results = []
        iteration = 0

        while iteration < self.max_tool_iterations:
            iteration += 1

            # Call LLM with current messages
            response = await self._call_llm(
                system=system,
                messages=messages,
                tools=available_tools,
            )

            # No tool calls - we're done
            if not response.tool_calls:
                # Update conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                if response.content:
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response.content
                    })

                # Limit history
                if len(self.conversation_history) > 40:
                    self.conversation_history = self.conversation_history[-40:]

                return AgentResponse(
                    content=response.content or "",
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    iterations=iteration,
                )

            # Has tool calls - execute them
            all_tool_calls.extend(response.tool_calls)

            # Execute tools and add results to messages
            tool_results_for_msg = []
            for tc in response.tool_calls:
                tool_name = tc["tool"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                logger.info(f"Tool call: {tool_name}({tool_args})")

                result = self.tool_executor.execute({
                    "tool": tool_name,
                    "args": tool_args
                })

                result_str = result.output if result.success else f"Error: {result.error}"
                logger.debug(f"Tool result: {result_str[:200]}...")

                all_tool_results.append({
                    "tool_call_id": tool_id,
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result_str,
                    "success": result.success,
                })

                tool_results_for_msg.append({
                    "tool_call_id": tool_id,
                    "content": result_str
                })

            # Build assistant message with tool calls
            assistant_msg = {"role": "assistant", "content": response.content}
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["tool"],
                            "arguments": json.dumps(tc["args"])
                        }
                    }
                    for tc in response.tool_calls
                ]
            messages.append(assistant_msg)

            # Add tool results to messages
            for tr in tool_results_for_msg:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": tr["content"]
                })

        # Max iterations reached
        logger.warning(f"Max tool iterations ({self.max_tool_iterations}) reached")

        fallback_msg = (
            "I've completed the tool operations "
            "but reached the iteration limit."
        )
        return AgentResponse(
            content=response.content or fallback_msg,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            iterations=iteration,
        )

    async def _call_llm(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> LLMResponse:
        """Call LLM with messages."""
        # Build full messages
        full_messages = [{"role": "system", "content": system}]
        full_messages.extend(messages)

        # Use OpenAI client directly
        client = self.llm_client._get_client()
        start = time.time()

        response = await client.chat.completions.create(
            model=self.llm_client.model,
            messages=full_messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.llm_client.temperature,
            max_tokens=self.llm_client.max_tokens,
        )

        choice = response.choices[0]
        message = choice.message

        # Extract content
        content = message.content or ""

        # Extract tool calls if present
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

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def process_sync(
        self,
        user_input: str,
        system_prompt: str = None,
        tools: list[dict] = None,
    ) -> AgentResponse:
        """Synchronous version of process."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.process(user_input, system_prompt, tools)
        )
