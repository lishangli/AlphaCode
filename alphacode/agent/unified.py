"""
Unified Agent for ALPHACODE.

A single agent that handles all conversation types.
LLM autonomously decides when to use MCTS exploration.

Key principle: Maximum flexibility - LLM makes all decisions.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from alphacode.llm.client import LLMClient, LLMResponse
from alphacode.llm.smart_cache import SmartCache
from alphacode.state.session_manager import SessionManager
from alphacode.tools.executor import TOOL_DEFINITIONS, ToolExecutor
from alphacode.utils.streaming_display import MCTSProgressDisplay, StreamingDisplay

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
    # Cache info
    from_cache: bool = False
    cache_type: str = ""
    # Performance metrics
    latency: float = 0.0
    tokens_used: int = 0
    # Clarification needed
    needs_clarification: bool = False
    clarification_question: str = ""


class UnifiedAgent:
    """
    Unified agent that handles all conversation types.

    LLM makes ALL decisions autonomously.
    No hardcoded rules about when to use tools.
    """

    # Minimal system prompt - no rules, let LLM decide
    SYSTEM_PROMPT = """You are ALPHACODE, a coding assistant.

Available tools: read, write, edit, bash, grep, glob, mcts_explore
"""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
        session_manager: SessionManager = None,
        max_tool_iterations: int = 5,
        enable_cache: bool = True,
        enable_clarification: bool = True,
        clarification_callback: Callable = None,
    ):
        """
        Initialize unified agent.

        Args:
            llm_client: LLM client for generation
            tool_executor: Tool executor for running tools
            session_manager: Optional session manager for Git tracking
            max_tool_iterations: Maximum tool call iterations
            enable_cache: Enable smart cache (LLM-driven)
            enable_clarification: Enable clarification requests
            clarification_callback: Callback for asking user clarification
        """
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.session_manager = session_manager
        self.max_tool_iterations = max_tool_iterations
        self.enable_clarification = enable_clarification
        self.clarification_callback = clarification_callback

        # Conversation history
        self.history: list[dict] = []

        # Current context for MCTS
        self.current_goal: str = None
        self.current_code: str = None

        # Smart cache (LLM-driven similarity judgment)
        self.cache: SmartCache = None
        if enable_cache:
            self.cache = SmartCache(
                llm_client=llm_client,
                ttl=3600,
                enable_llm_judgment=True,
            )
            self.cache.load()
            logger.info("Smart cache enabled")

        # Streaming display
        self.streaming_display: StreamingDisplay = None
        self.mcts_progress: MCTSProgressDisplay = None

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

                # Improved error message to help LLM understand
                if result.success:
                    result_str = result.output
                else:
                    result_str = (
                        f"Tool '{tool_name}' failed: {result.error}\n"
                        f"You can try a different approach or respond directly without tools."
                    )

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
            
            # Check if all tools failed and no content
            all_failed = all(not r.get("success", True) for r in all_tool_results)
            if all_failed and not response.content:
                # Add a hint for LLM to respond directly
                messages.append({
                    "role": "user",
                    "content": "Please respond directly without using any tools."
                })
                
                # Get direct response without tools
                final_response = await self._call_llm(
                    system=system,
                    messages=messages,
                    tools=None,  # No tools available
                )
                
                if final_response.content:
                    self.history.append({"role": "user", "content": user_input})
                    self.history.append({"role": "assistant", "content": final_response.content})
                    
                    return UnifiedResponse(
                        content=final_response.content,
                        tool_calls=all_tool_calls,
                        tool_results=all_tool_results,
                        iterations=iteration,
                        mcts_used=mcts_used,
                        mcts_result=mcts_result,
                    )

        # Max iterations reached
        logger.warning(f"Max iterations reached: {self.max_tool_iterations}")

        # If we have tool errors, let LLM respond directly
        error_messages = [
            r["result"] for r in all_tool_results 
            if not r.get("success", True)
        ]
        
        if error_messages and not response.content:
            # All tools failed, ask LLM to respond directly
            messages.append({
                "role": "user",
                "content": "The tool calls failed. Please respond directly without using tools."
            })
            
            final_response = await self._call_llm(
                system=system,
                messages=messages,
                tools=None,  # No tools, force direct response
            )
            
            return UnifiedResponse(
                content=final_response.content or "无法完成任务。",
                tool_calls=all_tool_calls,
                tool_results=all_tool_results,
                iterations=iteration,
                mcts_used=mcts_used,
                mcts_result=mcts_result,
            )

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

    # ========================================
    # New Features: Streaming, Cache, Clarification
    # ========================================

    async def process_stream(
        self,
        user_input: str,
        system_prompt: str = None,
        on_token: Callable[[str], None] = None,
        on_tool_call: Callable[[dict], None] = None,
        on_tool_result: Callable[[str], None] = None,
        on_mcts_progress: Callable[[dict], None] = None,
    ) -> AsyncIterator[str]:
        """
        Process user input with streaming output.

        LLM decides everything, we just stream the results.

        Args:
            user_input: User's message
            system_prompt: Optional custom system prompt
            on_token: Callback for each token
            on_tool_call: Callback for tool calls
            on_tool_result: Callback for tool results
            on_mcts_progress: Callback for MCTS progress

        Yields:
            str: Streamed content tokens
        """
        system = system_prompt or self.SYSTEM_PROMPT

        # Build messages
        messages = self.history.copy()
        messages.append({"role": "user", "content": user_input})

        # First, get LLM response (with potential tool calls)
        response = await self._call_llm(
            system=system,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        # Check for clarification request
        if self.enable_clarification and "[NEEDS_CLARIFICATION]" in (response.content or ""):
            question = response.content.replace("[NEEDS_CLARIFICATION]", "").strip()
            if on_token:
                on_token(f"\n❓ {question}\n")
            yield f"\n❓ {question}\n"
            return

        # If no tool calls, stream the response
        if not response.tool_calls:
            # Stream content
            if response.content:
                for char in response.content:
                    if on_token:
                        on_token(char)
                    yield char
                    await asyncio.sleep(0)  # Allow other tasks

            # Update history
            self.history.append({"role": "user", "content": user_input})
            if response.content:
                self.history.append({"role": "assistant", "content": response.content})
            return

        # Has tool calls - execute and show results
        # IMPORTANT: Don't show response.content when tool_calls exist
        # Some models return JSON description in content alongside tool_calls
        for tc in response.tool_calls:
            if on_tool_call:
                on_tool_call(tc)

        # Execute tools and show results
        tool_results = await self._execute_tools_streaming(
            response.tool_calls,
            on_mcts_progress=on_mcts_progress,
        )

        # Show tool execution results to user
        for tr in tool_results:
            result_text = tr["result"]
            if on_tool_result:
                if len(result_text) > 1000:
                    result_text = result_text[:1000] + "..."
                on_tool_result(result_text)
            # Yield result for streaming
            yield f"\n{result_text}\n"

        # Build assistant message with tool_calls
        assistant_msg = {"role": "assistant", "content": response.content or ""}
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
        for tr in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tr["id"],
                "content": tr["result"]
            })

        # Get streaming final response
        full_messages = [{"role": "system", "content": system}] + messages
        client = self.llm_client._get_client()
        stream = await client.chat.completions.create(
            model=self.llm_client.model,
            messages=full_messages,
            temperature=self.llm_client.temperature,
            max_tokens=self.llm_client.max_tokens,
            stream=True,
        )

        final_content = ""
        buffer = ""  # Buffer to detect JSON
        in_json = False
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                
                if not in_json:
                    # Check if token starts JSON tool call format
                    if '{' in token and ('"name"' in token or '"tool"' in token):
                        in_json = True
                        buffer = token
                        continue
                    # Normal content - output directly
                    final_content += token
                    if on_token:
                        on_token(token)
                    yield token
                else:
                    # Inside JSON - buffer until we see closing brace
                    buffer += token
                    # Check if JSON is complete (has matching braces)
                    if buffer.count('{') <= buffer.count('}'): 
                        in_json = False
                        buffer = ""  # Discard the JSON
                
                await asyncio.sleep(0)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        if final_content:
            self.history.append({"role": "assistant", "content": final_content})

    async def _execute_tools_streaming(
        self,
        tool_calls: list[dict],
        on_mcts_progress: Callable[[dict], None] = None,
    ) -> list[dict]:
        """
        Execute tools with streaming progress for MCTS.

        Args:
            tool_calls: List of tool calls
            on_mcts_progress: Callback for MCTS progress

        Returns:
            List of tool results
        """
        results = []

        for tc in tool_calls:
            tool_name = tc["tool"]
            tool_args = tc["args"]

            # For MCTS, show progress
            if tool_name == "mcts_explore" and on_mcts_progress:
                # Execute with progress callbacks
                result = await self._execute_mcts_with_progress(
                    tool_args,
                    on_progress=on_mcts_progress,
                )
            else:
                result = self.tool_executor.execute({
                    "tool": tool_name,
                    "args": tool_args,
                })

            result_str = result.output if result.success else f"Error: {result.error}"
            results.append({"id": tc["id"], "result": result_str})

        return results

    async def _execute_mcts_with_progress(
        self,
        args: dict,
        on_progress: Callable[[dict], None] = None,
    ) -> Any:
        """
        Execute MCTS with streaming progress updates.

        LLM decides the exploration, we just show progress.

        Args:
            args: MCTS arguments
            on_progress: Progress callback

        Returns:
            Execution result
        """
        # Import MCTS controller
        from alphacode.core.controller import MCTSController

        goal = args.get("goal", "")
        iterations = args.get("iterations", 20)

        # Create MCTS progress display
        progress_display = MCTSProgressDisplay(total_iterations=iterations)

        # Progress updates
        def update_progress(state: dict):
            progress_display.update(
                iteration=state.get("iteration", 0),
                best_score=state.get("best_score", 0),
                nodes_explored=state.get("nodes", 0),
                message=state.get("message", "")
            )
            if on_progress:
                on_progress(state)

        # Show initial state
        progress_display.show_thinking(f"探索: {goal[:50]}...")

        # Execute MCTS
        controller = MCTSController(
            llm_client=self.llm_client,
            sandbox_dir=self.tool_executor.sandbox_dir,
            max_iterations=iterations,
        )

        # Run with progress updates (simplified - actual MCTS needs modification)
        result = controller.run(goal)

        # Show completion
        if result:
            progress_display.complete(
                final_score=result.get("best_score", 0),
                total_nodes=result.get("nodes_explored", 0)
            )

            # Show best code preview
            best_code = result.get("best_code", "")
            if best_code:
                progress_display.show_best_code(
                    best_code,
                    result.get("best_score", 0)
                )

        # Return as tool result
        from alphacode.tools.executor import ExecutionResult
        return ExecutionResult(
            success=bool(result),
            output=result.get("summary", "探索完成") if result else "探索失败",
            data=result,
        )

    async def process_with_cache(
        self,
        user_input: str,
        system_prompt: str = None,
    ) -> UnifiedResponse:
        """
        Process with smart cache (LLM-driven similarity judgment).

        Cache decisions are made by LLM, not hardcoded rules.

        Args:
            user_input: User input
            system_prompt: Optional system prompt

        Returns:
            UnifiedResponse
        """
        if not self.cache:
            # No cache - normal processing
            return await self.process(user_input, system_prompt)

        # Try cache first
        async def compute():
            return await self.process(user_input, system_prompt)

        answer, from_cache, cache_meta = await self.cache.get_or_compute(
            query=user_input,
            compute_fn=compute,
            context=str(self.history[-5:]),  # Recent context
        )

        if from_cache:
            # Cache hit (LLM decided it's similar)
            return UnifiedResponse(
                content=answer,
                from_cache=True,
                cache_type=cache_meta.get("type", "unknown"),
            )

        # Cache miss - return computed result
        result = answer  # It's a UnifiedResponse
        result.from_cache = False
        result.cache_type = "computed"
        return result

    async def request_clarification(
        self,
        question: str,
    ) -> str:
        """
        Request clarification from user.

        Args:
            question: Clarification question

        Returns:
            User's clarification response
        """
        if self.clarification_callback:
            return await self.clarification_callback(question)

        # Default: return empty (CLI will handle it)
        return ""

    def save_cache(self):
        """Save cache to disk."""
        if self.cache:
            self.cache.save()
            logger.info("Cache saved")

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        if self.cache:
            return self.cache.stats()
        return {"enabled": False}
