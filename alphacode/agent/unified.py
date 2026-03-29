"""
Unified Agent for ALPHACODE.

A single agent that handles all conversation types.
LLM autonomously decides when to use MCTS exploration.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from alphacode.llm.client import LLMClient, LLMResponse
from alphacode.state.session_manager import SessionManager
from alphacode.tools.executor import TOOL_DEFINITIONS, ToolExecutor

logger = logging.getLogger(__name__)


@dataclass
class UnifiedResponse:
    """Response from unified agent."""
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    iterations: int = 0
    mcts_used: bool = False
    mcts_result: dict = None


class UnifiedAgent:
    """
    Unified agent that handles all conversation types.

    The LLM autonomously decides:
    - Simple questions: respond directly
    - Code exploration: use mcts_explore tool
    - File operations: use read/write/edit tools
    - Search: use grep/glob tools

    No hardcoded intent detection. LLM makes all decisions.
    """

    # System prompt for unified agent
    SYSTEM_PROMPT = """你是 ALPHACODE，一个智能代码助手。你可以帮助用户：
- 编写和优化代码
- 探索代码的不同实现方案
- 回答编程问题
- 执行文件操作和命令

**重要：判断用户意图后再决定是否使用工具**

不使用工具的情况（直接回复）：
- 问候："你好"、"hi"、"hello" → 回复问候
- 感谢："谢谢"、"thank you" → 回复"不客气"
- 简单问题："什么是快速排序？" → 直接解释
- 闲聊："今天天气怎么样？" → 自然对话

使用工具的情况：
- 读取文件："读取 config.py" → read 工具
- 写简单代码："写个 hello world" → write 工具
- 复杂算法："实现快速排序" → mcts_explore 工具
- 优化代码："优化这个函数" → mcts_explore 工具

工具说明：
- read/write/edit/bash/glob/grep：文件操作
- mcts_explore：复杂代码探索，返回最佳方案

规则：
- 简单问候不使用任何工具，直接回复
- 每个请求最多调用一次 mcts_explore
- 工具成功后直接展示结果，不重试"""

    # Shorter prompt for quick responses
    QUICK_SYSTEM_PROMPT = """你是 ALPHACODE，一个智能代码助手。

对于简单问题直接回答。对于代码任务，选择合适的工具：
- 简单代码 → write 工具
- 复杂算法 → mcts_explore 工具
- 文件操作 → read/write/edit 工具

直接行动，不要解释计划。"""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
        session_manager: SessionManager = None,
        max_tool_iterations: int = 5,
    ):
        """
        Initialize unified agent.

        Args:
            llm_client: LLM client for generation
            tool_executor: Tool executor for running tools
            session_manager: Optional session manager for Git tracking
            max_tool_iterations: Maximum tool call iterations
        """
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.session_manager = session_manager
        self.max_tool_iterations = max_tool_iterations

        # Conversation history
        self.history: list[dict] = []

        # Current context for MCTS
        self.current_goal: str = None
        self.current_code: str = None

    async def process(
        self,
        user_input: str,
        system_prompt: str = None,
    ) -> UnifiedResponse:
        """
        Process user input with autonomous tool calling.

        Args:
            user_input: User's message
            system_prompt: Optional custom system prompt

        Returns:
            UnifiedResponse with content and tool results
        """
        system = system_prompt or self.SYSTEM_PROMPT

        # Build messages with history
        messages = self.history.copy()
        messages.append({"role": "user", "content": user_input})

        all_tool_calls = []
        all_tool_results = []
        iteration = 0
        mcts_used = False
        mcts_result = None

        while iteration < self.max_tool_iterations:
            iteration += 1

            # Call LLM
            response = await self._call_llm(
                system=system,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )

            # No tool calls - done
            if not response.tool_calls:
                # Update history
                self.history.append({"role": "user", "content": user_input})
                if response.content:
                    self.history.append({"role": "assistant", "content": response.content})

                # Limit history length
                if len(self.history) > 40:
                    self.history = self.history[-40:]

                return UnifiedResponse(
                    content=response.content or "",
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    iterations=iteration,
                    mcts_used=mcts_used,
                    mcts_result=mcts_result,
                )

            # Has tool calls - execute them
            all_tool_calls.extend(response.tool_calls)

            # Check if MCTS is being used
            for tc in response.tool_calls:
                if tc["tool"] == "mcts_explore":
                    mcts_used = True

            # Execute tools
            tool_results_for_msg = []
            for tc in response.tool_calls:
                tool_name = tc["tool"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                logger.info(f"Tool call: {tool_name}({list(tool_args.keys())})")

                result = self.tool_executor.execute({
                    "tool": tool_name,
                    "args": tool_args
                })

                result_str = result.output if result.success else f"Error: {result.error}"

                # Store MCTS result if applicable
                if tool_name == "mcts_explore" and result.success and result.data:
                    mcts_result = result.data.get("mcts_result")
                    # Update current context
                    self.current_goal = tool_args.get("goal")
                    self.current_code = result.data.get("best_code")

                all_tool_results.append({
                    "tool_call_id": tool_id,
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result_str,
                    "success": result.success,
                    "data": result.data if result.success else None,
                })

                tool_results_for_msg.append({
                    "tool_call_id": tool_id,
                    "content": result_str
                })

            # Build assistant message
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

            # Add tool results
            for tr in tool_results_for_msg:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": tr["content"]
                })

        # Max iterations reached
        logger.warning(f"Max iterations reached: {self.max_tool_iterations}")

        return UnifiedResponse(
            content=response.content or "操作完成，但达到迭代上限。",
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            iterations=iteration,
            mcts_used=mcts_used,
            mcts_result=mcts_result,
        )

    async def _call_llm(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> LLMResponse:
        """Call LLM with messages and tools."""
        full_messages = [{"role": "system", "content": system}]
        full_messages.extend(messages)

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

        content = message.content or ""

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

    def process_sync(
        self,
        user_input: str,
        system_prompt: str = None,
    ) -> UnifiedResponse:
        """Synchronous version of process."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.process(user_input, system_prompt)
        )

    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        self.current_goal = None
        self.current_code = None

    def get_context(self) -> dict:
        """Get current context for continuation."""
        return {
            "goal": self.current_goal,
            "code": self.current_code,
            "history": self.history[-10:],  # Last 10 exchanges
        }

    def record_to_session(self, user_input: str, response: UnifiedResponse):
        """Record exchange to session if available."""
        if not self.session_manager:
            return

        # Record user message
        self.session_manager.record_event(
            event_type="user_message",
            content=user_input,
        )

        # Record tool calls
        for tc in response.tool_calls:
            self.session_manager.record_tool_call(
                tool=tc["tool"],
                args=tc["args"],
                result=None,  # Will be updated
            )

        # Record assistant response
        self.session_manager.record_event(
            event_type="assistant_message",
            content=response.content,
        )
