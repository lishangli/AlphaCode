"""
MCTS Selector module.

Handles node selection using UCB (Upper Confidence Bound) algorithm.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

from alphacode.config import MCTSConfig
from alphacode.core.node import MCTSNode, NodeStatus

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Result of node selection."""
    node: MCTSNode
    path: list[MCTSNode]
    ucb_score: float


class UCBSelector:
    """
    UCB-based node selector.

    Selects nodes to expand using Upper Confidence Bound algorithm.
    Balances exploration vs exploitation.

    UCB formula:
    UCB = exploitation + c * sqrt(log(parent_visits) / node_visits)

    Where:
    - exploitation = node.value_avg
    - c = exploration_weight (sqrt(2) by default)
    """

    def __init__(self, config: MCTSConfig):
        """
        Initialize selector.

        Args:
            config: MCTS configuration
        """
        self.config = config
        self.exploration_weight = config.exploration_weight
        self.exploration_decay = config.exploration_decay

    def select(
        self,
        tree: Any,  # SearchTree
        start_node: MCTSNode = None,
    ) -> SelectionResult:
        """
        Select a node to expand.

        Traverses the tree from root (or start_node) using UCB
        until reaching a leaf node or a node that needs expansion.

        Args:
            tree: Search tree
            start_node: Optional starting node (default: root)

        Returns:
            SelectionResult with selected node and path
        """
        # Start from root or given node
        current = start_node or tree.get_root()
        if current is None:
            return None

        path = [current]
        iteration = tree.total_iterations if hasattr(tree, 'total_iterations') else 0

        # Traverse down using UCB
        while True:
            children = tree.get_children(current.id)
            valid_children = [
                c for c in children
                if c.status != NodeStatus.PRUNED
            ]

            # No valid children - this is our selection
            if not valid_children:
                break

            # Find child with highest UCB
            best_child = None
            best_ucb = float('-inf')

            for child in valid_children:
                ucb = self._calculate_ucb(child, current, iteration)
                child.ucb_score = ucb

                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child

            if best_child is None:
                # All children are invalid
                break

            current = best_child
            path.append(current)

            # Check depth limit
            if current.depth >= self.config.max_depth:
                break

        return SelectionResult(
            node=current,
            path=path,
            ucb_score=current.ucb_score,
        )

    def _calculate_ucb(
        self,
        node: MCTSNode,
        parent: MCTSNode,
        iteration: int = 0,
    ) -> float:
        """
        Calculate UCB score for a node.

        Args:
            node: Node to calculate UCB for
            parent: Parent node
            iteration: Current iteration (for decay)

        Returns:
            UCB score
        """
        # Unvisited nodes have infinite UCB
        if node.visits == 0:
            return float('inf')

        # Calculate exploration weight with decay
        c = self.exploration_weight * (self.exploration_decay ** iteration)

        # Exploitation term
        exploitation = node.value_avg

        # Exploration term
        exploration = c * math.sqrt(
            math.log(max(parent.visits, 1)) / node.visits
        )

        return exploitation + exploration

    def select_for_expansion(
        self,
        tree: Any,
        max_depth: int = None,
    ) -> MCTSNode | None:
        """
        Select a node that can be expanded.

        A node can be expanded if:
        - It's a leaf node (no children)
        - It hasn't reached max depth
        - It's not pruned

        Args:
            tree: Search tree
            max_depth: Maximum depth for expansion

        Returns:
            Node to expand, or None
        """
        result = self.select(tree)
        if result is None:
            return None

        node = result.node
        max_d = max_depth or self.config.max_depth

        # Check if expandable
        if node.depth >= max_d:
            return None

        if node.status == NodeStatus.PRUNED:
            return None

        return node


class IslandSelector:
    """
    Island-based selection strategy.

    Manages multiple independent search islands,
    selecting which island to explore next.
    """

    def __init__(self, num_islands: int = 4):
        """
        Initialize island selector.

        Args:
            num_islands: Number of islands
        """
        self.num_islands = num_islands
        self.current_island = 0
        self.island_visits = [0] * num_islands

    def select_island(self) -> int:
        """
        Select next island to explore.

        Uses round-robin selection.

        Returns:
            Island index
        """
        island = self.current_island
        self.current_island = (self.current_island + 1) % self.num_islands
        self.island_visits[island] += 1
        return island

    def get_island_stats(self) -> list[dict]:
        """Get statistics for each island."""
        return [
            {
                "island_id": i,
                "visits": self.island_visits[i],
            }
            for i in range(self.num_islands)
        ]

    def reset(self):
        """Reset island selector."""
        self.current_island = 0
        self.island_visits = [0] * self.num_islands


class EpsilonGreedySelector:
    """
    Epsilon-greedy node selector.

    With probability epsilon, selects randomly (exploration).
    Otherwise, selects best UCB (exploitation).
    """

    def __init__(self, config: MCTSConfig, epsilon: float = 0.1):
        """
        Initialize epsilon-greedy selector.

        Args:
            config: MCTS configuration
            epsilon: Exploration probability
        """
        self.ucb_selector = UCBSelector(config)
        self.epsilon = epsilon

    def select(self, tree: Any) -> SelectionResult:
        """
        Select a node using epsilon-greedy strategy.

        Args:
            tree: Search tree

        Returns:
            SelectionResult
        """
        import random

        if random.random() < self.epsilon:
            # Random selection for exploration
            return self._select_random(tree)
        else:
            # UCB selection for exploitation
            return self.ucb_selector.select(tree)

    def _select_random(self, tree: Any) -> SelectionResult:
        """Select a random expandable node."""
        import random

        # Get all nodes that can be expanded
        expandable = []
        for node in tree.nodes.values():
            if (node.status != NodeStatus.PRUNED and
                node.depth < self.config.max_depth):
                expandable.append(node)

        if not expandable:
            return None

        node = random.choice(expandable)
        path = tree.get_path_to_root(node.id)

        return SelectionResult(
            node=node,
            path=path,
            ucb_score=0.0,
        )
