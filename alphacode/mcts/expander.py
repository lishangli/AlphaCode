"""
MCTS Expander module.

Handles node expansion using LLM to generate actions.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from alphacode.config import MCTSConfig
from alphacode.core.node import Action, MCTSNode
from alphacode.llm.client import LLMClient
from alphacode.llm.prompts import PromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class ExpansionResult:
    """Result of node expansion."""
    actions: list[Action]
    reasoning: str = ""
    confidence: float = 0.5
    entropy: float = 0.5


class LLMExpander:
    """
    LLM-driven node expander.

    Uses LLM to generate improvement actions for code.
    Supports multi-branch expansion (generating multiple actions).
    """

    def __init__(
        self,
        config: MCTSConfig,
        llm_client: LLMClient = None,
    ):
        """
        Initialize expander.

        Args:
            config: MCTS configuration
            llm_client: LLM client for generation
        """
        self.config = config
        self.llm_client = llm_client
        self.prompt_builder = PromptBuilder()

        # Statistics
        self.expansions = 0
        self.total_actions = 0

    async def expand(
        self,
        node: MCTSNode,
        goal: str,
        tree: Any = None,  # SearchTree
        num_actions: int = None,
    ) -> ExpansionResult:
        """
        Expand a node by generating improvement actions.

        Args:
            node: Node to expand
            goal: User goal
            tree: Search tree (for inspirations)
            num_actions: Number of actions to generate

        Returns:
            ExpansionResult with generated actions
        """
        self.expansions += 1

        # Determine number of actions
        n_actions = num_actions or self.config.num_actions_per_expand

        # Use self-assessment to decide number of branches
        if self.expansions > 1 and self.llm_client:
            n_actions = await self._determine_branches(node, goal)

        # Generate actions
        if self.llm_client:
            actions = await self._generate_llm_actions(node, goal, tree, n_actions)
        else:
            actions = self._generate_simple_actions(node, goal, n_actions)

        self.total_actions += len(actions)

        return ExpansionResult(
            actions=actions,
            confidence=0.5,  # Would be set by LLM response
        )

    async def _determine_branches(
        self,
        node: MCTSNode,
        goal: str,
    ) -> int:
        """
        Determine number of branches using self-assessment.

        High confidence → single branch
        Low confidence → multi branch

        Args:
            node: Current node
            goal: User goal

        Returns:
            Number of branches (1 or 2)
        """
        prompt = f"""Rate your confidence (0.0-1.0) for solving this task.

Goal: {goal[:100]}

Current code state:
```
{node.code[:200]}
```

Consider:
- Is the goal clear and specific?
- Do you know the solution pattern?
- Is the current code on the right track?

Respond with just a number (e.g., "0.9" or "0.3")."""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=10,
            )

            import re
            match = re.search(r'(\d+\.?\d*)', response.content)
            if match:
                confidence = float(match.group(1))

                logger.info(
                    f"Self-assessment: confidence={confidence:.3f}, "
                    f"entropy={response.entropy:.3f}"
                )

                # High confidence + low entropy → single branch
                if confidence > 0.8 and response.entropy < 0.05:
                    return 1

        except Exception as e:
            logger.warning(f"Self-assessment failed: {e}")

        return 2  # Default to multi-branch

    async def _generate_llm_actions(
        self,
        node: MCTSNode,
        goal: str,
        tree: Any,
        num_actions: int,
    ) -> list[Action]:
        """
        Generate actions using LLM.

        Args:
            node: Node to expand
            goal: User goal
            tree: Search tree
            num_actions: Number of actions to generate

        Returns:
            List of generated actions
        """
        # Get inspirations from tree
        inspirations = self._get_inspirations(node, tree) if tree else []

        # Get previous attempts
        previous = self._get_previous_attempts(node, tree) if tree else []

        # Build prompt
        prompt = self.prompt_builder.build_expand_prompt(
            goal=goal,
            current_code=node.code,
            node=node,
            inspirations=inspirations,
            previous_attempts=previous,
            artifacts=node.artifacts,
            num_actions=num_actions,
        )

        try:
            response = await self.llm_client.generate_json(
                prompt=prompt["user"],
                system=prompt["system"],
                temperature=0.7,
            )

            # Parse actions
            actions = []
            for action_data in response.get("actions", []):
                action = Action(
                    id=str(uuid.uuid4())[:8],
                    description=action_data.get("description", ""),
                    reasoning=action_data.get("reasoning", ""),
                    tool_calls=action_data.get("tool_calls", []),
                )
                action.confidence = action_data.get("confidence", 0.5)
                actions.append(action)

            # Pad with simple actions if needed
            if len(actions) < num_actions:
                simple = self._generate_simple_actions(node, goal, num_actions - len(actions))
                actions.extend(simple)

            return actions[:num_actions]

        except Exception as e:
            logger.warning(f"LLM expansion failed: {e}")
            return self._generate_simple_actions(node, goal, num_actions)

    def _generate_simple_actions(
        self,
        node: MCTSNode,
        goal: str,
        num_actions: int,
    ) -> list[Action]:
        """
        Generate simple actions without LLM.

        Fallback when LLM is not available.

        Args:
            node: Node to expand
            goal: User goal
            num_actions: Number of actions to generate

        Returns:
            List of simple actions
        """
        actions = []

        # Generate code based on goal keywords
        code = self._generate_code_from_goal(goal)

        actions.append(Action(
            id=str(uuid.uuid4())[:8],
            description="Implement solution",
            reasoning="Direct implementation based on goal keywords",
            confidence=0.6,
            tool_calls=[{
                "tool": "write",
                "args": {
                    "path": "program.py",
                    "content": code
                }
            }]
        ))

        # Add variation
        if num_actions > 1:
            actions.append(Action(
                id=str(uuid.uuid4())[:8],
                description="Alternative implementation",
                reasoning="Alternative approach",
                confidence=0.5,
                tool_calls=[{
                    "tool": "edit",
                    "args": {
                        "path": "program.py",
                        "old": "def ",
                        "new": '"""Function docstring."""\ndef '
                    }
                }]
            ))

        return actions[:num_actions]

    def _generate_code_from_goal(self, goal: str) -> str:
        """Generate simple code based on goal keywords."""
        import re

        goal_lower = goal.lower()

        # Check for specific patterns
        if "fibonacci" in goal_lower or "斐波那契" in goal_lower:
            return """def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""

        if "sort" in goal_lower or "排序" in goal_lower:
            return """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
"""

        if "prime" in goal_lower or "质数" in goal_lower:
            return """def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
"""

        # Extract function/class name
        func_match = re.search(r'(?:def|function|函数)\s+(\w+)', goal)
        if func_match:
            func_name = func_match.group(1)
            return f"""def {func_name}():
    # Implementation for: {goal[:50]}
    pass
"""

        # Default template
        return f"""# Implementation for: {goal[:50]}
def solve():
    pass
"""

    def _get_inspirations(
        self,
        node: MCTSNode,
        tree: Any,
    ) -> list[MCTSNode]:
        """Get inspiration nodes from feature grid."""
        inspirations = []

        if not tree or not hasattr(tree, 'feature_grid'):
            return inspirations

        # Get diverse nodes from feature grid
        exclude_coords = None
        if node.feature_coords:
            exclude_coords = node.feature_coords.to_key()

        diverse_ids = tree.feature_grid.get_diverse_nodes(
            exclude_coords=exclude_coords,
            n=self.config.num_inspirations // 2,
        )

        for nid in diverse_ids:
            n = tree.get_node(nid)
            if n:
                inspirations.append(n)

        # Add best node if available
        if tree.best_node_id:
            best = tree.get_node(tree.best_node_id)
            if best and best.id != node.id:
                inspirations.append(best)

        return inspirations[:self.config.num_inspirations]

    def _get_previous_attempts(
        self,
        node: MCTSNode,
        tree: Any,
    ) -> list[dict]:
        """Get previous attempts from path to root."""
        attempts = []

        if not tree:
            return attempts

        path = tree.get_path_to_root(node.id)

        for n in path[1:4]:  # Skip self, get up to 3 ancestors
            if n.action:
                attempts.append({
                    "description": n.action.description,
                    "error": n.action.error,
                    "score": n.value_avg,
                })

        return attempts

    def get_stats(self) -> dict:
        """Get expander statistics."""
        return {
            "expansions": self.expansions,
            "total_actions": self.total_actions,
            "avg_actions_per_expansion": (
                self.total_actions / self.expansions
                if self.expansions > 0 else 0
            ),
        }


class RandomExpander:
    """
    Random action expander for testing.
    """

    def __init__(self, config: MCTSConfig):
        self.config = config

    def expand(
        self,
        node: MCTSNode,
        goal: str,
        tree: Any = None,
        num_actions: int = None,
    ) -> ExpansionResult:
        """Generate random actions."""
        import random

        n = num_actions or self.config.num_actions_per_expand
        actions = []

        for i in range(n):
            actions.append(Action(
                id=str(uuid.uuid4())[:8],
                description=f"Random action {i+1}",
                confidence=random.uniform(0.3, 0.8),
                tool_calls=[{
                    "tool": "edit",
                    "args": {
                        "path": "program.py",
                        "old": "#",
                        "new": f"# Modified {i+1}\n#"
                    }
                }]
            ))

        return ExpansionResult(actions=actions)
