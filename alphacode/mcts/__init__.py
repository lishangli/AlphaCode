"""
MCTS (Monte Carlo Tree Search) module.

Contains the core MCTS components:
- Selector: UCB-based node selection
- Expander: LLM-driven node expansion
- Evaluator: Code evaluation with caching
- Backpropagator: Value backpropagation
"""

from alphacode.mcts.evaluator import (
    Backpropagator,
    EvaluationCache,
    LightEvaluator,
    ParallelEvaluator,
)
from alphacode.mcts.expander import (
    ExpansionResult,
    LLMExpander,
    RandomExpander,
)
from alphacode.mcts.selector import (
    EpsilonGreedySelector,
    IslandSelector,
    SelectionResult,
    UCBSelector,
)

__all__ = [
    # Selector
    "UCBSelector",
    "IslandSelector",
    "EpsilonGreedySelector",
    "SelectionResult",
    # Expander
    "LLMExpander",
    "RandomExpander",
    "ExpansionResult",
    # Evaluator
    "EvaluationCache",
    "LightEvaluator",
    "ParallelEvaluator",
    "Backpropagator",
]
