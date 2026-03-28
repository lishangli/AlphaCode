"""
MCTS Evaluator module.

Handles code evaluation with caching and parallel evaluation support.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from alphacode.config import MCTSConfig
from alphacode.core.node import EvaluationResult, MCTSNode
from alphacode.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Evaluation cache entry."""
    result: EvaluationResult
    timestamp: float
    code_hash: str
    goal_hash: str


class EvaluationCache:
    """
    Evaluation result cache.

    Caches evaluation results to avoid redundant LLM calls
    for the same code + goal combination.
    """

    def __init__(self, cache_dir: str = None, ttl: int = 3600):
        """
        Initialize cache.

        Args:
            cache_dir: Cache directory path
            ttl: Cache TTL in seconds
        """
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "alphacode",
            "evaluations"
        )
        self.ttl = ttl

        # In-memory cache
        self._cache: dict[str, CacheEntry] = {}

        # Ensure directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Stats
        self.hits = 0
        self.misses = 0

    def _hash(self, code: str, goal: str) -> str:
        """Generate cache key."""
        data = f"{code}|{goal}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def get(self, code: str, goal: str) -> EvaluationResult | None:
        """
        Get cached evaluation result.

        Args:
            code: Code to evaluate
            goal: User goal

        Returns:
            Cached EvaluationResult or None
        """
        key = self._hash(code, goal)

        # Check memory cache
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry.timestamp < self.ttl:
                self.hits += 1
                logger.debug(f"Cache hit: {key}")
                return entry.result

        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    data = json.load(f)

                entry = CacheEntry(
                    result=EvaluationResult.from_dict(data["result"]),
                    timestamp=data["timestamp"],
                    code_hash=data["code_hash"],
                    goal_hash=data["goal_hash"],
                )

                if time.time() - entry.timestamp < self.ttl:
                    self._cache[key] = entry
                    self.hits += 1
                    return entry.result

            except Exception as e:
                logger.debug(f"Cache read error: {e}")

        self.misses += 1
        return None

    def set(self, code: str, goal: str, result: EvaluationResult):
        """
        Save evaluation result to cache.

        Args:
            code: Evaluated code
            goal: User goal
            result: Evaluation result
        """
        key = self._hash(code, goal)

        entry = CacheEntry(
            result=result,
            timestamp=time.time(),
            code_hash=hashlib.sha256(code.encode()).hexdigest()[:16],
            goal_hash=hashlib.sha256(goal.encode()).hexdigest()[:16],
        )

        # Save to memory
        self._cache[key] = entry

        # Save to disk
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_file, "w") as f:
                json.dump({
                    "result": result.to_dict(),
                    "timestamp": entry.timestamp,
                    "code_hash": entry.code_hash,
                    "goal_hash": entry.goal_hash,
                }, f)
        except Exception as e:
            logger.debug(f"Cache write error: {e}")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
            "entries": len(self._cache),
        }

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0


class LightEvaluator:
    """
    Lightweight evaluator for MCTS search.

    Fast evaluation without heavy LLM calls.
    Used during MCTS iterations.
    """

    def __init__(self, config: MCTSConfig):
        self.config = config

    def evaluate(self, code: str, goal: str) -> EvaluationResult:
        """
        Quick evaluation of code.

        Checks:
        1. Syntax validity
        2. Code quality heuristics
        3. Goal relevance

        Args:
            code: Code to evaluate
            goal: User goal

        Returns:
            EvaluationResult
        """
        result = EvaluationResult()

        # 1. Syntax check
        syntax_score = self._check_syntax(code)
        result.metrics["syntax"] = syntax_score

        if syntax_score < 0.1:
            result.score = 0.0
            return result

        # 2. Code quality
        quality_score = self._check_quality(code)
        result.metrics["quality"] = quality_score

        # 3. Goal relevance
        relevance_score = self._check_relevance(code, goal)
        result.metrics["relevance"] = relevance_score

        # Calculate score
        result.score = (
            0.2 * syntax_score +
            0.3 * quality_score +
            0.5 * relevance_score
        )

        return result

    def _check_syntax(self, code: str) -> float:
        """Check if code has valid syntax."""
        import ast

        try:
            ast.parse(code)
            return 1.0
        except SyntaxError:
            return 0.0

    def _check_quality(self, code: str) -> float:
        """Check code quality heuristics."""
        if not code or len(code.strip()) < 10:
            return 0.0

        score = 0.3

        # Check for function definitions
        if "def " in code:
            score += 0.2

        # Check for class definitions
        if "class " in code:
            score += 0.2

        # Check for return statements
        if "return " in code:
            score += 0.1

        # Check for proper indentation
        lines = code.split("\n")
        indented = [
            line for line in lines
            if line.startswith("    ") or line.startswith("\t")
        ]
        if len(indented) > 2:
            score += 0.2

        return min(score, 1.0)

    def _check_relevance(self, code: str, goal: str) -> float:
        """Check if code is relevant to goal."""
        goal_lower = goal.lower()
        code_lower = code.lower()

        # Concept keywords with Chinese translations
        concepts = {
            "fibonacci": ["fibonacci", "fib", "斐波那契"],
            "sort": ["sort", "sorted", "排序"],
            "prime": ["prime", "is_prime", "质数", "素数"],
            "search": ["search", "find", "查找"],
            "tree": ["tree", "node", "树"],
            "list": ["list", "array", "列表"],
            "stack": ["stack", "push", "pop", "栈"],
            "queue": ["queue", "队列"],
            "graph": ["graph", "vertex", "图"],
        }

        # Find concepts in goal
        matched = []
        for concept, keywords in concepts.items():
            for kw in keywords:
                if kw in goal_lower:
                    matched.append((concept, keywords))
                    break

        if not matched:
            return 0.5  # No specific concept

        # Check if code mentions keywords
        score = 0.0
        for _concept, keywords in matched:
            for kw in keywords:
                if kw in code_lower:
                    score += 1.0 / len(matched)
                    break

        return min(score, 1.0)


class ParallelEvaluator:
    """
    Parallel evaluator with caching.

    Runs syntax, quality, and LLM evaluations in parallel.
    """

    def __init__(
        self,
        config: MCTSConfig,
        llm_client: LLMClient = None,
        cache: EvaluationCache = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.cache = cache or EvaluationCache()
        self.light_evaluator = LightEvaluator(config)

    async def evaluate(
        self,
        code: str,
        goal: str,
        use_cache: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate code with parallel execution.

        Args:
            code: Code to evaluate
            goal: User goal
            use_cache: Whether to use cache

        Returns:
            EvaluationResult
        """
        # Check cache
        if use_cache:
            cached = self.cache.get(code, goal)
            if cached:
                return cached

        # Light evaluation first
        light_result = self.light_evaluator.evaluate(code, goal)

        # If light score is very low, skip heavy evaluation
        if light_result.score < 0.2:
            self.cache.set(code, goal, light_result)
            return light_result

        # Parallel heavy evaluation
        if self.llm_client:
            heavy_result = await self._heavy_evaluate(code, goal, light_result)
            self.cache.set(code, goal, heavy_result)
            return heavy_result

        self.cache.set(code, goal, light_result)
        return light_result

    async def _heavy_evaluate(
        self,
        code: str,
        goal: str,
        light_result: EvaluationResult,
    ) -> EvaluationResult:
        """Run heavy LLM evaluation in parallel."""
        # Create tasks
        quality_task = self._evaluate_quality_llm(code)
        progress_task = self._evaluate_progress_llm(code, goal)

        # Run in parallel
        try:
            quality_result, progress_result = await asyncio.gather(
                quality_task, progress_task,
                return_exceptions=True
            )

            # Merge results
            result = EvaluationResult()
            result.metrics = light_result.metrics.copy()

            if not isinstance(quality_result, Exception):
                result.metrics["quality_llm"] = quality_result

            if not isinstance(progress_result, Exception):
                result.metrics["progress_llm"] = progress_result

            # Calculate final score
            result.score = (
                0.1 * result.metrics.get("syntax", 0) +
                0.15 * result.metrics.get("quality", 0) +
                0.25 * result.metrics.get("relevance", 0) +
                0.25 * result.metrics.get("quality_llm", 0.5) +
                0.25 * result.metrics.get("progress_llm", 0.5)
            )

            return result

        except Exception as e:
            logger.warning(f"Parallel evaluation failed: {e}")
            return light_result

    async def _evaluate_quality_llm(self, code: str) -> float:
        """Evaluate code quality using LLM."""
        prompt = f"""Rate code quality (0.0-1.0). Reply with just the number.

Code:
```
{code[:500]}
```

Score:"""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=10,
            )

            import re
            match = re.search(r'(\d+\.?\d*)', response.content)
            if match:
                return float(match.group(1))

        except Exception as e:
            logger.debug(f"Quality LLM eval failed: {e}")

        return 0.5

    async def _evaluate_progress_llm(self, code: str, goal: str) -> float:
        """Evaluate progress toward goal using LLM."""
        prompt = f"""Rate progress toward goal (0.0-1.0). Reply with just the number.

Goal: {goal[:100]}

Code:
```
{code[:500]}
```

Score:"""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=10,
            )

            import re
            match = re.search(r'(\d+\.?\d*)', response.content)
            if match:
                return float(match.group(1))

        except Exception as e:
            logger.debug(f"Progress LLM eval failed: {e}")

        return 0.5

    def evaluate_sync(
        self,
        code: str,
        goal: str,
        use_cache: bool = True,
    ) -> EvaluationResult:
        """Synchronous evaluation."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.evaluate(code, goal, use_cache)
        )


class Backpropagator:
    """
    MCTS backpropagation handler.

    Updates node statistics from leaf to root.
    """

    def __init__(self, config: MCTSConfig):
        self.config = config

    def backpropagate(
        self,
        node: MCTSNode,
        value: float,
        tree: Any,  # SearchTree
    ):
        """
        Backpropagate value through tree.

        Updates:
        - Visit counts
        - Value sums
        - Best values

        Args:
            node: Leaf node
            value: Evaluation value
            tree: Search tree
        """
        # Get path to root
        path = tree.get_path_to_root(node.id)

        # Update each node in path
        for n in path:
            n.update_stats(value)

        # Update island best
        if node.island_id < len(tree.islands):
            island = tree.islands[node.island_id]
            island.total_visits += 1
            island.update_best(node)

        # Update global best
        if node.value_avg > tree.best_value:
            tree.best_value = node.value_avg
            tree.best_node_id = node.id

    def backpropagate_with_decay(
        self,
        node: MCTSNode,
        value: float,
        tree: Any,
        decay_factor: float = 0.95,
    ):
        """
        Backpropagate with depth-based decay.

        Values from deeper nodes are weighted less.

        Args:
            node: Leaf node
            value: Evaluation value
            tree: Search tree
            decay_factor: Decay per level
        """
        path = tree.get_path_to_root(node.id)

        for i, n in enumerate(path):
            decayed_value = value * (decay_factor ** i)
            n.update_stats(decayed_value)

        # Update best
        if node.value_avg > tree.best_value:
            tree.best_value = node.value_avg
            tree.best_node_id = node.id
