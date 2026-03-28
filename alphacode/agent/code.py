"""
Code Generation Agent for ALPHACODE.

Handles code generation tasks that will be optimized by MCTS.
"""

import logging

from alphacode.agent.base import AgentResponse, BaseAgent
from alphacode.llm.client import LLMClient
from alphacode.state.session_manager import SessionManager
from alphacode.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


class CodeAgent(BaseAgent):
    """
    Code generation agent.

    Generates initial code solutions that MCTS will optimize.
    Always writes to program.py for MCTS to process.
    """

    SYSTEM_PROMPT = """You are ALPHACODE, a code generation assistant.

You have access to tools for creating and modifying code.

IMPORTANT RULES:
1. ALWAYS write code to `program.py` - this is where MCTS will optimize it
2. Create complete, working implementations
3. Include proper function definitions and test code if appropriate

Example tool usage:
- write(path="program.py", content="def solve():\\n    ...")
- edit(path="program.py", old="old code", new="new code")

Do NOT use other file names like fibonacci.py or main.py. Use program.py only.

After writing code, MCTS will explore variations to find the best solution."""

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

    async def process(
        self,
        user_input: str,
        system_prompt: str = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Process code generation request.

        Args:
            user_input: User's code request
            system_prompt: Optional custom system prompt

        Returns:
            AgentResponse with generated code
        """
        # Start code generation session
        if not self.current_session:
            self.start_session(intent="code_generation", goal=user_input)

        # Record user request
        self.record_message("user", user_input)

        # Generate initial solution
        response = await self._generate_initial(user_input, system_prompt)

        # Record result
        self.record_message("assistant", response.content)

        return response

    async def generate_initial(
        self,
        goal: str,
        system_prompt: str = None,
    ) -> AgentResponse:
        """
        Generate initial code solution.

        This is the main entry point for code generation.
        The generated code will be optimized by MCTS.

        Args:
            goal: Code generation goal
            system_prompt: Optional custom system prompt

        Returns:
            AgentResponse with generated code in program.py
        """
        # Start session
        if not self.current_session:
            self.start_session(intent="code_generation", goal=goal)

        return await self._generate_initial(goal, system_prompt)

    async def _generate_initial(
        self,
        goal: str,
        system_prompt: str = None,
    ) -> AgentResponse:
        """Internal implementation of initial code generation."""
        import json

        messages = [{"role": "user", "content": goal}]
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
                return AgentResponse(
                    content=response.content or "",
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    iterations=iteration,
                    session_id=self.current_session.session_id if self.current_session else "",
                )

            # Execute tools
            all_tool_calls.extend(response.tool_calls)

            # Build assistant message
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

            # Execute tools
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

                # Add to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_str,
                })

        return AgentResponse(
            content=response.content or "",
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            iterations=iteration,
            session_id=self.current_session.session_id if self.current_session else "",
        )

    async def improve_code(
        self,
        current_code: str,
        goal: str,
        feedback: str = None,
    ) -> AgentResponse:
        """
        Improve existing code based on feedback.

        Args:
            current_code: Current code state
            goal: Original goal
            feedback: Optional feedback/guidance

        Returns:
            AgentResponse with improved code
        """
        prompt = f"""Improve the following code for the goal.

Goal: {goal}

Current code:
```
{current_code[:1000]}
```

{f"Feedback: {feedback}" if feedback else ""}

Make improvements and write the updated code to program.py."""

        return await self._generate_initial(prompt)
