"""
Search tree and MAP-Elites feature grid.
"""

import logging
import random
from dataclasses import dataclass, field

from alphacode.core.node import MCTSNode

logger = logging.getLogger(__name__)


@dataclass
class FeatureGrid:
    """
    MAP-Elites feature grid.

    Maintains diverse solutions across feature dimensions.
    Each cell contains the best node for that feature combination.
    """
    dimensions: list[str] = field(
        default_factory=lambda: ["complexity", "approach", "quality_tier"]
    )
    bins_per_dim: int = 10

    # Grid storage: coords_key -> node_id
    grid: dict[str, str] = field(default_factory=dict)

    # Reverse index: node_id -> coords_key
    node_to_coords: dict[str, str] = field(default_factory=dict)

    def get_cell(self, node: MCTSNode) -> str | None:
        """Get the grid cell key for a node."""
        if node.feature_coords:
            return node.feature_coords.to_key()
        return None

    def add(self, node: MCTSNode) -> bool:
        """
        Add node to grid.

        Returns True if added (either new cell or replaced existing).
        """
        key = self.get_cell(node)
        if key is None:
            return False

        existing_id = self.grid.get(key)
        if existing_id is None:
            # Empty cell, add directly
            self.grid[key] = node.id
            self.node_to_coords[node.id] = key
            return True

        return False  # Need comparison with existing

    def add_if_better(self, node: MCTSNode, existing_node: MCTSNode) -> bool:
        """
        Replace existing if node is better.

        Returns True if replaced.
        """
        key = self.get_cell(node)
        if key is None:
            return False

        if node.value_avg > existing_node.value_avg:
            # Remove old mapping
            if existing_node.id in self.node_to_coords:
                del self.node_to_coords[existing_node.id]

            # Add new
            self.grid[key] = node.id
            self.node_to_coords[node.id] = key
            return True

        return False

    def get_diverse_nodes(
        self,
        exclude_coords: str | None = None,
        n: int = 3
    ) -> list[str]:
        """
        Get nodes from different feature regions.

        Used for inspiration sampling.
        """
        nodes = []
        for key, node_id in self.grid.items():
            if key != exclude_coords:
                nodes.append(node_id)
            if len(nodes) >= n:
                break
        return nodes

    def get_random_cells(self, n: int = 3) -> list[str]:
        """Get random cell keys."""
        keys = list(self.grid.keys())
        return random.sample(keys, min(n, len(keys)))

    def coverage(self) -> float:
        """Calculate grid coverage ratio."""
        total_cells = self.bins_per_dim ** len(self.dimensions)
        return len(self.grid) / total_cells if total_cells > 0 else 0.0

    def clear(self):
        """Clear the grid."""
        self.grid.clear()
        self.node_to_coords.clear()


@dataclass
class Island:
    """
    Single island in the island-based evolution model.

    Each island is a separate sub-population that explores independently.
    Periodic migration between islands maintains diversity.
    """
    id: int = 0
    root_id: str = ""

    # Nodes in this island
    nodes: set[str] = field(default_factory=set)

    # Statistics
    total_visits: int = 0
    best_node_id: str | None = None
    best_value: float = 0.0

    # Exploration preference
    exploration_weight: float = 1.0
    focus_area: str | None = None

    def update_best(self, node: MCTSNode):
        """Update best node if this node is better."""
        if node.value_avg > self.best_value:
            self.best_value = node.value_avg
            self.best_node_id = node.id

    def add_node(self, node_id: str):
        """Add a node to this island."""
        self.nodes.add(node_id)

    def remove_node(self, node_id: str):
        """Remove a node from this island."""
        self.nodes.discard(node_id)
        if self.best_node_id == node_id:
            self.best_node_id = None
            self.best_value = 0.0

    def size(self) -> int:
        """Get number of nodes in island."""
        return len(self.nodes)


@dataclass
class SearchTree:
    """
    Complete MCTS search tree.

    Contains:
    - All nodes
    - Feature grid (MAP-Elites)
    - Multiple islands
    """
    # ========== Session Info ==========
    session_id: str = ""
    goal: str = ""

    # ========== Node Storage ==========
    nodes: dict[str, MCTSNode] = field(default_factory=dict)
    root_id: str | None = None

    # ========== Feature Grid ==========
    feature_grid: FeatureGrid = field(default_factory=FeatureGrid)

    # ========== Islands ==========
    islands: list[Island] = field(default_factory=list)
    num_islands: int = 4
    current_island_idx: int = 0

    # ========== Migration ==========
    migration_interval: int = 50
    migration_rate: float = 0.1
    last_migration_iteration: int = 0

    # ========== Global Statistics ==========
    total_iterations: int = 0
    best_node_id: str | None = None
    best_value: float = 0.0

    # ========== UCB Parameters ==========
    exploration_weight: float = 1.41

    def get_node(self, node_id: str) -> MCTSNode | None:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def add_node(self, node: MCTSNode):
        """
        Add node to search tree.

        Updates:
        - Node storage
        - Island membership
        - Feature grid
        - Global best
        """
        # Store node
        self.nodes[node.id] = node

        # Update island
        if node.island_id < len(self.islands):
            self.islands[node.island_id].add_node(node.id)
            self.islands[node.island_id].update_best(node)

        # Update feature grid
        if node.feature_coords:
            key = node.feature_coords.to_key()
            existing_id = self.feature_grid.grid.get(key)

            if existing_id is None:
                self.feature_grid.grid[key] = node.id
                self.feature_grid.node_to_coords[node.id] = key
            else:
                existing = self.nodes.get(existing_id)
                if existing and node.value_avg > existing.value_avg:
                    # Replace
                    if existing_id in self.feature_grid.node_to_coords:
                        del self.feature_grid.node_to_coords[existing_id]
                    self.feature_grid.grid[key] = node.id
                    self.feature_grid.node_to_coords[node.id] = key

        # Update global best
        if node.value_avg > self.best_value:
            self.best_value = node.value_avg
            self.best_node_id = node.id

    def remove_node(self, node_id: str):
        """Remove node from search tree."""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # Remove from island
        if node.island_id < len(self.islands):
            self.islands[node.island_id].remove_node(node_id)

        # Remove from feature grid
        if node_id in self.feature_grid.node_to_coords:
            key = self.feature_grid.node_to_coords[node_id]
            del self.feature_grid.node_to_coords[node_id]
            if self.feature_grid.grid.get(key) == node_id:
                del self.feature_grid.grid[key]

        # Remove from parent's children
        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if parent and node_id in parent.children_ids:
                parent.children_ids.remove(node_id)

        # Remove from storage
        del self.nodes[node_id]

        # Update global best if needed
        if self.best_node_id == node_id:
            self._recalculate_best()

    def _recalculate_best(self):
        """Recalculate global best node."""
        self.best_node_id = None
        self.best_value = 0.0

        for node in self.nodes.values():
            if node.value_avg > self.best_value:
                self.best_value = node.value_avg
                self.best_node_id = node.id

    def get_children(self, node_id: str) -> list[MCTSNode]:
        """Get children of a node."""
        node = self.get_node(node_id)
        if node is None:
            return []
        return [
            self.nodes[cid]
            for cid in node.children_ids
            if cid in self.nodes
        ]

    def get_path_to_root(self, node_id: str) -> list[MCTSNode]:
        """Get path from node to root."""
        path = []
        current = self.get_node(node_id)

        while current:
            path.append(current)
            if current.parent_id:
                current = self.get_node(current.parent_id)
            else:
                break

        return path

    def get_root(self) -> MCTSNode | None:
        """Get root node."""
        if self.root_id:
            return self.get_node(self.root_id)
        return None

    def init_islands(self, root_node_id: str):
        """Initialize islands with root node."""
        self.islands = []
        for i in range(self.num_islands):
            island = Island(
                id=i,
                root_id=root_node_id,
            )
            island.add_node(root_node_id)
            self.islands.append(island)

    def get_next_island(self) -> Island:
        """Get next island in round-robin fashion."""
        island = self.islands[self.current_island_idx]
        self.current_island_idx = (self.current_island_idx + 1) % self.num_islands
        return island

    def size(self) -> int:
        """Get total number of nodes."""
        return len(self.nodes)

    def depth(self) -> int:
        """Get maximum depth of tree."""
        if not self.nodes:
            return 0
        return max(node.depth for node in self.nodes.values())

    def get_stats(self) -> dict:
        """Get tree statistics."""
        return {
            "total_nodes": self.size(),
            "max_depth": self.depth(),
            "total_iterations": self.total_iterations,
            "best_value": self.best_value,
            "grid_coverage": self.feature_grid.coverage(),
            "islands": [
                {
                    "id": i.id,
                    "size": i.size(),
                    "best_value": i.best_value,
                }
                for i in self.islands
            ],
        }
