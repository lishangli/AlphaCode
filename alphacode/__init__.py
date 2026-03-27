"""
MCTS-Agent: MCTS-based code exploration agent with Git state management.
"""

from alphacode.config import MCTSConfig
from alphacode.core.controller import MCTSController
from alphacode.core.node import MCTSNode, NodeStatus, Action, EvaluationResult
from alphacode.core.tree import SearchTree, FeatureGrid, Island

__version__ = "0.1.0"

__all__ = [
    "MCTSConfig",
    "MCTSController", 
    "MCTSNode",
    "NodeStatus",
    "Action",
    "EvaluationResult",
    "SearchTree",
    "FeatureGrid",
    "Island",
]