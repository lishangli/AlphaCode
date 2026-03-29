"""
MCTS Explore Tool for ALPHACODE.

Allows LLM to autonomously trigger MCTS exploration for code optimization.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphacode.config import MCTSConfig
    from alphacode.core.controller import MCTSController, Solution

logger = logging.getLogger(__name__)


@dataclass
class MCTSExploreResult:
    """Result of MCTS exploration."""
    success: bool
    best_code: str = ""
    best_score: float = 0.0
    alternatives: list[dict] = field(default_factory=list)
    iterations: int = 0
    nodes_explored: int = 0
    test_passed: bool = False
    message: str = ""
    error: str = ""


class MCTSExploreTool:
    """
    MCTS exploration tool for code optimization.

    This tool wraps MCTSController to provide a clean interface for LLM
    to trigger code exploration and optimization.
    """

    # Tool definition for LLM function calling
    DEFINITION = {
        "type": "function",
        "function": {
            "name": "mcts_explore",
            "description": """探索和优化代码的多种实现方案。

适用于：
- 复杂算法问题（排序、搜索、动态规划等）
- 需要性能优化的代码
- 不确定最佳实现方式的情况
- 用户要求"优化"、"试试其他方法"

不适用于：
- 简单的一行代码
- 配置文件、文档
- 已确定实现方式的简单任务

返回最佳代码方案和备选方案。""",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "代码目标描述，如'实现快速排序算法'或'优化斐波那契函数性能'"
                    },
                    "initial_code": {
                        "type": "string",
                        "description": "可选：初始代码，如果不提供则从头探索"
                    },
                    "iterations": {
                        "type": "integer",
                        "description": "探索迭代次数，默认10，复杂问题可增加到20-30",
                        "default": 10
                    },
                    "focus": {
                        "type": "string",
                        "enum": ["performance", "readability", "correctness", "balanced"],
                        "description": (
                    "优化重点：performance=性能, "
                    "readability=可读性, correctness=正确性, balanced=平衡"
                ),
                        "default": "balanced"
                    },
                    "test_cases": {
                        "type": "string",
                        "description": "可选：测试用例，用于验证代码正确性",
                    }
                },
                "required": ["goal"]
            }
        }
    }

    def __init__(
        self,
        config: MCTSConfig = None,
        working_dir: str = None,
    ):
        """
        Initialize MCTS explore tool.

        Args:
            config: MCTS configuration
            working_dir: Working directory for code files
        """
        self.config = config or MCTSConfig()
        self.working_dir = working_dir or os.getcwd()
        self.controller: MCTSController = None
        self.last_solution: Solution = None

    def execute(
        self,
        goal: str,
        initial_code: str = None,
        iterations: int = 10,
        focus: str = "balanced",
        test_cases: str = None,
    ) -> MCTSExploreResult:
        """
        Execute MCTS exploration.

        Args:
            goal: Code goal description
            initial_code: Optional starting code
            iterations: Number of exploration iterations
            focus: Optimization focus
            test_cases: Optional test cases

        Returns:
            MCTSExploreResult with best code and alternatives
        """
        logger.info(
            f"Starting MCTS exploration: goal='{goal}', "
            f"iterations={iterations}, focus={focus}"
        )

        try:
            # Import dependencies lazily to avoid circular imports
            from alphacode.config import MCTSConfig
            from alphacode.core.controller import MCTSController

            # Create temporary working directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Initialize git in temp directory for MCTS
                import subprocess
                subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
                subprocess.run(
                    ["git", "config", "user.email", "alphacode@example.com"],
                    cwd=tmpdir, capture_output=True
                )
                subprocess.run(
                    ["git", "config", "user.name", "ALPHACODE"],
                    cwd=tmpdir, capture_output=True
                )

                # Write initial code if provided
                code_file = os.path.join(tmpdir, "program.py")
                if initial_code:
                    with open(code_file, 'w') as f:
                        f.write(initial_code)
                    logger.info(f"Wrote initial code ({len(initial_code)} bytes)")

                # Create config for this exploration
                if self.config:
                    explore_config = MCTSConfig(
                        max_iterations=iterations,
                        max_depth=self.config.max_depth,
                        exploration_weight=self.config.exploration_weight,
                        llm=self.config.llm,
                    )
                else:
                    explore_config = MCTSConfig(max_iterations=iterations)

                # Change to temp directory for controller
                old_dir = os.getcwd()
                os.chdir(tmpdir)

                try:
                    # Create controller
                    self.controller = MCTSController(config=explore_config)

                    # Run exploration using solve method
                    solution = self.controller.solve(
                        goal=goal,
                        initial_code=initial_code or "",
                    )
                    self.last_solution = solution
                finally:
                    # Restore original directory
                    os.chdir(old_dir)

                # Build result
                if solution and solution.best_node:
                    best_code = solution.best_node.code or ""
                    best_score = solution.best_node.value_avg

                    # Collect alternatives from search tree
                    alternatives = []
                    if solution.search_tree and solution.search_tree.nodes:
                        # Get all nodes except the best one
                        all_nodes = list(solution.search_tree.nodes.values())
                        sorted_nodes = sorted(
                            all_nodes,
                            key=lambda n: n.value_avg,
                            reverse=True
                        )
                        for node in sorted_nodes[1:4]:  # Skip best, take next 3
                            if node.code and node.code != best_code:
                                alternatives.append({
                                    "code": node.code,
                                    "score": node.value_avg,
                                    "description": (
                                        node.action.description
                                        if node.action
                                        else "探索方案"
                                    ),
                                })

                    # Build message
                    nodes_count = solution.search_tree.size() if solution.search_tree else 1
                    message = self._build_message(
                        goal=goal,
                        best_score=best_score,
                        iterations=solution.iterations,
                        nodes=nodes_count,
                        test_passed=solution.test_passed,
                        has_alternatives=len(alternatives) > 0,
                    )

                    return MCTSExploreResult(
                        success=True,
                        best_code=best_code,
                        best_score=best_score,
                        alternatives=alternatives,
                        iterations=solution.iterations,
                        nodes_explored=nodes_count,
                        test_passed=solution.test_passed,
                        message=message,
                    )
                else:
                    return MCTSExploreResult(
                        success=False,
                        message="探索未能找到有效解决方案",
                        error="No solution found",
                    )

        except Exception as e:
            logger.error(f"MCTS exploration failed: {e}")
            return MCTSExploreResult(
                success=False,
                message=f"探索过程出错: {str(e)}",
                error=str(e),
            )

    def _build_message(
        self,
        goal: str,
        best_score: float,
        iterations: int,
        nodes: int,
        test_passed: bool,
        has_alternatives: bool,
    ) -> str:
        """Build user-friendly message."""
        parts = [f"✓ 探索完成: {goal}"]

        # Quality assessment
        if best_score >= 0.8:
            parts.append(f"  评分: {best_score:.2f} (优秀)")
        elif best_score >= 0.6:
            parts.append(f"  评分: {best_score:.2f} (良好)")
        else:
            parts.append(f"  评分: {best_score:.2f} (可改进)")

        # Stats
        parts.append(f"  探索了 {nodes} 个节点，{iterations} 次迭代")

        # Test status
        if test_passed:
            parts.append("  ✓ 测试通过")

        # Alternatives hint
        if has_alternatives:
            parts.append("  输入 /alternatives 查看其他方案")

        return "\n".join(parts)

    def continue_exploration(
        self,
        additional_iterations: int = 10,
    ) -> MCTSExploreResult:
        """
        Continue exploration from last solution.

        Args:
            additional_iterations: More iterations to run

        Returns:
            Updated MCTSExploreResult
        """
        if not self.controller or not self.last_solution:
            return MCTSExploreResult(
                success=False,
                message="没有之前的探索可以继续",
                error="No previous exploration",
            )

        # Continue from best node
        # Note: This would require modifying MCTSController to support continuation
        # For now, return a message about limitation
        return MCTSExploreResult(
            success=False,
            message="继续探索功能开发中，请重新发起探索请求",
            error="Continuation not implemented",
        )

    def get_alternatives(self) -> list[dict]:
        """Get alternative solutions from last exploration."""
        if not self.last_solution or not self.last_solution.search_tree:
            return []

        best_code = (
            self.last_solution.best_node.code
            if self.last_solution.best_node
            else ""
        )

        alternatives = []
        all_nodes = list(self.last_solution.search_tree.nodes.values())
        sorted_nodes = sorted(
            all_nodes,
            key=lambda n: n.value_avg,
            reverse=True
        )

        for i, node in enumerate(sorted_nodes):
            if node.code and node.code != best_code:
                alternatives.append({
                    "index": len(alternatives) + 1,
                    "code": node.code,
                    "score": node.value_avg,
                    "description": (
                        node.action.description
                        if node.action
                        else f"方案 {i+1}"
                    ),
                })

        return alternatives[:5]  # Return top 5 alternatives


# Convenience function for tool registration
def get_mcts_tool_definition() -> dict:
    """Get tool definition for LLM function calling."""
    return MCTSExploreTool.DEFINITION
