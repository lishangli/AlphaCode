"""
Conversation Agent for ALPHACODE.

Handles general conversation, questions, and chitchat.
"""

import json
import logging

from alphacode.agent.base import AgentResponse, BaseAgent
from alphacode.llm.client import LLMClient
from alphacode.state.session_manager import SessionManager
from alphacode.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


class ConversationAgent(BaseAgent):
    """
    Conversation agent for general interactions.

    Handles:
    - Questions about code
    - File exploration
    - General chitchat
    """

    SYSTEM_PROMPT = """You are ALPHACODE, a helpful AI coding assistant.

You can help with:
- Answering programming questions
- Exploring and explaining code
- Writing and modifying code
- Running tests and commands

RULES FOR TOOL USAGE:
1. For greetings like "hello", "你好", "hi" - respond WITHOUT tools, just chat naturally
2. For simple questions like "what is X?" - respond WITHOUT tools, explain directly
3. For code-related requests like "read file", "show me X.py" - USE read/glob tools
4. For code generation like "write a function" - USE write/edit tools

Available tools: read, write, edit, glob, grep, bash

Do NOT try to read files like "ALPHACODE.md" or "README.md" unless user explicitly asks.

Be friendly and helpful. For chitchat, respond naturally without tools."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
        session_manager: SessionManager = None,
        max_tool_iterations: int = 5,
    ):
        super().__init__(
            llm_client=llm_client,
            tool_executor=tool_executor,
            session_manager=session_manager,
            max_tool_iterations=max_tool_iterations,
        )
        self.conversation_history: list[dict] = []

    async def process(
        self,
        user_input: str,
        system_prompt: str = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Process user input with tool calling loop.

        Args:
            user_input: User's message
            system_prompt: Optional custom system prompt

        Returns:
            AgentResponse
        """
        # Ensure session exists
        if not self.current_session:
            self.start_session(intent="conversation")

        # Record user message
        self.record_message("user", user_input)

        # Build messages with history
        messages = self._build_messages(user_input)

        all_tool_calls = []
        all_tool_results = []
        iteration = 0

        while iteration < self.max_tool_iterations:
            iteration += 1

            # Call LLM
            response = await self._call_llm_with_tools(
                user_input="",
                system_prompt=system_prompt or self.SYSTEM_PROMPT,
                messages=messages,
            )

            # No tool calls - done
            if not response.tool_calls:
                # Record assistant response
                self.record_message("assistant", response.content)

                # Update history
                self._update_history(user_input, response.content)

                return AgentResponse(
                    content=response.content or "",
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    iterations=iteration,
                    session_id=self.current_session.session_id if self.current_session else "",
                )

            # Execute tools
            all_tool_calls.extend(response.tool_calls)

            # Build assistant message with tool calls
            assistant_msg = {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
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
            }
            messages.append(assistant_msg)

            # Execute each tool
            for tc in response.tool_calls:
                tool_name = tc["tool"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                logger.info(f"Tool call: {tool_name}({tool_args})")

                success, result_str = await self._execute_tool_call(tool_name, tool_args)

                # Record tool call
                self.record_tool_call(tool_name, tool_args, result_str, success)

                all_tool_results.append({
                    "tool_call_id": tool_id,
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result_str,
                    "success": success,
                })

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_str,
                })

        # Max iterations reached
        return AgentResponse(
            content=response.content or "I've completed the operations.",
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            iterations=iteration,
            session_id=self.current_session.session_id if self.current_session else "",
        )

    def _build_messages(self, user_input: str) -> list[dict]:
        """Build messages with conversation history."""
        messages = []

        # Add recent history (last 10 turns)
        for msg in self.conversation_history[-20:]:
            messages.append(msg)

        # Add current input
        messages.append({"role": "user", "content": user_input})

        return messages

    def _update_history(self, user_input: str, assistant_response: str):
        """Update conversation history."""
        self.conversation_history.append({"role": "user", "content": user_input})
        if assistant_response:
            self.conversation_history.append({"role": "assistant", "content": assistant_response})

        # Limit history
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-40:]

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
