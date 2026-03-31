"""
Progressive MCTS with real-time output.

Key principle: LLM decides everything. We only provide technical infrastructure
for streaming output, parsing, and progress display.

NO hardcoded prompts that guide LLM's decisions.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

from alphacode.llm.client import LLMClient
from alphacode.utils.streaming_display import MCTSProgressDisplay

logger = logging.getLogger(__name__)


@dataclass
class ExplorationProgress:
    """Progress update from MCTS exploration."""
    iteration: int
    total_iterations: int
    best_score: float
    best_code: Optional[str]
    nodes_explored: int
    status: str  # "running", "improved", "stagnant", "completed"
    message: str = ""
    
    elapsed_time: float = 0.0
    iteration_time: float = 0.0
    
    def is_improved(self) -> bool:
        """Check if this iteration improved the best score."""
        return self.status == "improved"


@dataclass
class ExplorationResult:
    """Final result from MCTS exploration."""
    best_code: str
    best_score: float
    total_iterations: int
    total_nodes: int
    total_time: float
    exploration_history: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dict."""
        return {
            "best_code": self.best_code,
            "best_score": self.best_score,
            "total_iterations": self.total_iterations,
            "total_nodes": self.total_nodes,
            "total_time": self.total_time,
            "summary": f"探索完成: {self.total_iterations}次迭代, "
                       f"最优分数{self.best_score:.2f}, "
                       f"{self.total_nodes}个节点",
        }


class ProgressiveMCTS:
    """
    MCTS with progressive output.
    
    Design principle:
    - LLM decides exploration strategy, prompts, and evaluation
    - We only provide: streaming output, progress display, code parsing
    - NO hardcoded prompts or decision guidance
    
    Usage:
        # Let LLM decide the prompt
        prompt = user_goal  # Or whatever LLM decides
        
        async for progress in mcts.explore_stream(
            prompt=prompt,
            generate_fn=llm_generate,  # LLM generates
            evaluate_fn=llm_evaluate,  # LLM evaluates
        ):
            if progress.is_improved():
                show_code(progress.best_code)
    """
    
    def __init__(
        self,
        max_iterations: int = 20,
    ):
        """
        Initialize progressive MCTS.
        
        Args:
            max_iterations: Maximum exploration iterations
        """
        self.max_iterations = max_iterations
        
        # Progress tracking
        self._best_score = 0.0
        self._best_code = ""
        self._nodes_explored = 0
        self._start_time = 0.0
        self._history: list[dict] = []
    
    async def explore_stream(
        self,
        generate_fn: Callable[[], AsyncIterator[str]],
        evaluate_fn: Callable[[str], float],
        on_progress: Callable[[ExplorationProgress], None] = None,
    ) -> AsyncIterator[ExplorationProgress]:
        """
        Explore with streaming progress updates.
        
        LLM provides generate_fn and evaluate_fn - we don't decide anything.
        
        Args:
            generate_fn: Async function that yields candidate code
            evaluate_fn: Function that evaluates code and returns score
            on_progress: Optional progress callback
            
        Yields:
            ExplorationProgress updates
        """
        self._start_time = time.time()
        self._best_score = 0.0
        self._best_code = ""
        self._nodes_explored = 0
        self._history = []
        
        for iteration in range(self.max_iterations):
            iteration_start = time.time()
            
            # LLM generates candidates
            candidates = []
            async for code in generate_fn():
                if code and len(code) > 10:
                    candidates.append(code)
            
            # Evaluate candidates (LLM decides how to evaluate)
            improved = False
            for code in candidates:
                score = evaluate_fn(code)
                self._nodes_explored += 1
                
                if score > self._best_score:
                    self._best_score = score
                    self._best_code = code
                    improved = True
                    
                    self._history.append({
                        "iteration": iteration,
                        "score": score,
                        "improved": True,
                    })
            
            # Determine status
            status = "improved" if improved else "stagnant"
            if iteration == self.max_iterations - 1:
                status = "completed"
            
            # Build progress
            progress = ExplorationProgress(
                iteration=iteration + 1,
                total_iterations=self.max_iterations,
                best_score=self._best_score,
                best_code=self._best_code if improved else None,
                nodes_explored=self._nodes_explored,
                status=status,
                message=f"迭代 {iteration + 1}: {'发现更好方案' if improved else '继续探索'}",
                elapsed_time=time.time() - self._start_time,
                iteration_time=time.time() - iteration_start,
            )
            
            if on_progress:
                on_progress(progress)
            
            yield progress
            
            # Early stopping if very good score (but LLM can override)
            if self._best_score >= 0.95:
                logger.info(f"Early stopping: score {self._best_score:.2f}")
                break
        
        # Final progress
        final_progress = ExplorationProgress(
            iteration=self.max_iterations,
            total_iterations=self.max_iterations,
            best_score=self._best_score,
            best_code=self._best_code,
            nodes_explored=self._nodes_explored,
            status="completed",
            message="探索完成",
            elapsed_time=time.time() - self._start_time,
        )
        
        yield final_progress
    
    async def explore(
        self,
        generate_fn: Callable[[], AsyncIterator[str]],
        evaluate_fn: Callable[[str], float],
        show_progress: bool = True,
    ) -> ExplorationResult:
        """
        Explore and return final result.
        
        Args:
            generate_fn: Async generator yielding candidate code
            evaluate_fn: Function evaluating code -> score
            show_progress: Show progress display
            
        Returns:
            ExplorationResult
        """
        display = None
        if show_progress:
            display = MCTSProgressDisplay(total_iterations=self.max_iterations)
        
        final_iteration = 0
        
        async for progress in self.explore_stream(generate_fn, evaluate_fn):
            final_iteration = progress.iteration
            
            if display:
                display.update(
                    iteration=progress.iteration,
                    best_score=progress.best_score,
                    nodes_explored=progress.nodes_explored,
                    message=progress.message,
                )
                
                if progress.is_improved() and progress.best_code:
                    display.show_best_code(
                        progress.best_code,
                        progress.best_score,
                    )
        
        if display:
            display.complete(self._best_score, self._nodes_explored)
        
        return ExplorationResult(
            best_code=self._best_code,
            best_score=self._best_score,
            total_iterations=final_iteration,
            total_nodes=self._nodes_explored,
            total_time=time.time() - self._start_time,
            exploration_history=self._history,
        )
    
    def get_exploration_history(self) -> list[dict]:
        """Get exploration history."""
        return self._history


# Helper function to parse code from LLM output (pure utility, no decisions)
def parse_code_blocks(content: str) -> list[str]:
    """
    Parse code blocks from LLM output.
    
    This is a pure utility function - no decisions, just parsing.
    
    Args:
        content: LLM output content
        
    Returns:
        List of code blocks found
    """
    import re
    
    code_blocks = []
    
    # Find ```python blocks
    pattern = r"```(?:python)?\s*(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        code = match.strip()
        if code and len(code) > 10:
            code_blocks.append(code)
    
    return code_blocks