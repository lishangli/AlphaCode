"""
Node selector for MCTS-Agent.

Implements UCB-based node selection.
"""

import logging
import math

from alphacode.config import MCTSConfig
from alphacode.core.node import MCTSNode, NodeStatus
from alphacode.core.tree import Island, SearchTree

logger = logging.getLogger(__name__)


class NodeSelector:
    """
    Node selector using UCB.

    Strategies:
    - UCB1: Standard upper confidence bound
    - UCB-Tuned: Variance-aware selection
    - epsilon-greedy: Random exploration
    """

    def __init__(self, config: MCTSConfig):
        self.config = config
        self.iteration = 0

    def select(
        self,
        tree: SearchTree,
        island: Island = None
    ) -> MCTSNode | None:
        """
        Select a node for expansion.

        Args:
            tree: Search tree
            island: Optional island to select from

        Returns:
            Selected node or None
        """
        if island:
            return self._select_from_island(tree, island)

        # Default: select from current island
        current_island = tree.islands[tree.current_island_idx]
        return self._select_from_island(tree, current_island)

    def _select_from_island(
        self,
        tree: SearchTree,
        island: Island
    ) -> MCTSNode | None:
        """Select node from a specific island."""
        # Start from island root
        current = tree.get_node(island.root_id)

        if current is None:
            return None

        # Traverse down using UCB
        while True:
            children = tree.get_children(current.id)

            # Filter valid children
            valid_children = [
                c for c in children
                if c.status != NodeStatus.PRUNED
            ]

            if not valid_children:
                # Leaf node
                return current

            # Select best child
            best_child = self._select_child(current, valid_children)

            if best_child is None:
                return current

            current = best_child

    def _select_child(
        self,
        parent: MCTSNode,
        children: list[MCTSNode]
    ) -> MCTSNode | None:
        """Select best child using UCB."""
        best_score = float('-inf')
        best_child = None

        for child in children:
            score = self.calculate_ucb(child, parent)
            child.ucb_score = score

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def calculate_ucb(
        self,
        node: MCTSNode,
        parent: MCTSNode
    ) -> float:
        """
        Calculate UCB score.

        UCB = exploitation + c * sqrt(ln(N) / n)

        Args:
            node: Node to calculate score for
            parent: Parent node

        Returns:
            UCB score
        """
        if node.visits == 0:
            return float('inf')

        # Dynamic exploration weight
        c = self._get_exploration_weight()

        # Exploitation term
        exploitation = node.value_avg

        # Exploration term
        exploration = c * math.sqrt(
            math.log(max(parent.visits, 1)) / node.visits
        )

        return exploitation + exploration

    def _get_exploration_weight(self) -> float:
        """Get current exploration weight (with decay)."""
        return self.config.exploration_weight * (
            self.config.exploration_decay ** self.iteration
        )

    def update_iteration(self, iteration: int):
        """Update current iteration."""
        self.iteration = iteration


class EpsilonGreedySelector(NodeSelector):
    """
    Epsilon-greedy node selector.

    With probability epsilon, select random node.
    Otherwise, select best by value.
    """

    def __init__(self, config: MCTSConfig, epsilon: float = 0.1):
        super().__init__(config)
        self.epsilon = epsilon

    def _select_child(
        self,
        parent: MCTSNode,
        children: list[MCTSNode]
    ) -> MCTSNode | None:
        """Select child using epsilon-greedy."""
        import random

        if random.random() < self.epsilon:
            # Random selection
            return random.choice(children)

        # Greedy selection by value
        return max(children, key=lambda c: c.value_avg)
