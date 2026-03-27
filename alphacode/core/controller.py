"""
MCTS Controller for MCTS-Agent.

Main controller that orchestrates the MCTS search process.
"""

import os
import sys
import uuid
import time
import math
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from alphacode.config import MCTSConfig
from alphacode.core.node import MCTSNode, NodeStatus, Action, EvaluationResult, FeatureCoords
from alphacode.core.tree import SearchTree, FeatureGrid, Island
from alphacode.state.git_manager import GitStateManager
from alphacode.tools.executor import ToolExecutor, ToolResult
from alphacode.evaluation.evaluator import CascadeEvaluator
from alphacode.llm.client import LLMClient
from alphacode.llm.prompts import PromptBuilder
from alphacode.llm.intent import IntentDetector, IntentType, IntentResult, get_response_for_intent

logger = logging.getLogger(__name__)


@dataclass
class Solution:
    """
    Solution result from MCTS search.
    """
    session_id: str
    goal: str
    best_code: str
    best_score: float
    best_node: Optional[MCTSNode]
    search_tree: SearchTree
    iterations: int
    total_nodes: int
    test_passed: bool = False
    test_output: str = ""
    
    def get_alternative_solutions(self, n: int = 5) -> List[MCTSNode]:
        """Get alternative solutions from feature grid."""
        alternatives = []
        
        for node_id in self.search_tree.feature_grid.grid.values():
            if node_id != self.best_node.id if self.best_node else True:
                node = self.search_tree.get_node(node_id)
                if node:
                    alternatives.append(node)
        
        alternatives.sort(key=lambda n: n.value_avg, reverse=True)
        return alternatives[:n]
    
    def get_report(self) -> str:
        """Generate exploration report."""
        lines = [
            f"# MCTS Exploration Report",
            f"",
            f"## Session: {self.session_id}",
            f"## Goal: {self.goal}",
            f"",
            f"## Statistics",
            f"- Total iterations: {self.iterations}",
            f"- Total nodes explored: {self.total_nodes}",
            f"- Best score: {self.best_score:.3f}",
            f"- Test passed: {self.test_passed}",
            f"",
            f"## Best Solution (score: {self.best_score:.3f})",
            f"```",
            f"{self.best_code[:500]}{'...' if len(self.best_code) > 500 else ''}",
            f"```",
        ]
        
        return "\n".join(lines)


class MCTSController:
    """
    MCTS Controller.
    
    Optimized flow:
    1. Intent check - filter non-code tasks
    2. MCTS search with 2 branches
    3. Smart execution based on confidence
    4. Light evaluation during search
    5. Final test at the end
    """
    
    def __init__(self, config: MCTSConfig):
        self.config = config
        
        # Core components
        self.git_manager = GitStateManager(branch_prefix=config.git_branch_prefix)
        self.tool_executor = ToolExecutor(root_path=self.git_manager.root_path)
        self.evaluator = CascadeEvaluator(config=config, llm_client=None)
        self.prompt_builder = PromptBuilder()
        self.llm_client: Optional[LLMClient] = None
        self.intent_detector: Optional[IntentDetector] = None
        
        # Search state
        self.search_tree: Optional[SearchTree] = None
        self.session_id: str = ""
        self.iteration: int = 0
        self.running: bool = False
        
        # History for convergence detection
        self.score_history: List[float] = []
        
        # Retry counter
        self.retry_count: int = 0
    
    def _init_llm(self):
        """Initialize LLM client."""
        if self.config.llm.api_key:
            self.llm_client = LLMClient(
                api_key=self.config.llm.api_key,
                api_base=self.config.llm.api_base,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                timeout=self.config.llm.timeout,
            )
            self.evaluator.llm_client = self.llm_client
            self.intent_detector = IntentDetector(self.llm_client)
    
    def check_intent(self, user_input: str) -> IntentResult:
        """
        Check if user input is a code task.
        
        Returns:
            IntentResult with classification
        """
        if not self.config.intent_check or not self.intent_detector:
            # Skip intent check, assume code task
            return IntentResult(
                intent=IntentType.CODE_TASK,
                confidence=1.0,
                reason="Intent check disabled",
                code_hint=user_input,
            )
        
        logger.info(f"Checking intent for: {user_input[:50]}...")
        return self.intent_detector.detect_sync(user_input)
    
    def solve(
        self,
        goal: str,
        initial_code: str = "",
        working_dir: str = None,
    ) -> Solution:
        """
        Main entry point: solve a programming task.
        """
        # Setup working directory
        if working_dir:
            os.chdir(working_dir)
            self.git_manager.root_path = working_dir
        
        # Initialize LLM
        self._init_llm()
        
        # Initialize session
        self._init_session(goal, initial_code)
        
        # Main MCTS loop
        self.running = True
        no_improvement_count = 0
        
        while self.running and self.iteration < self.config.max_iterations:
            try:
                old_best = self.search_tree.best_value
                self._run_iteration()
                self.iteration += 1
                
                # Check for no improvement
                if self.search_tree.best_value <= old_best:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                
                # Early stop if no improvement
                if no_improvement_count >= 3:
                    logger.info(f"No improvement for {no_improvement_count} iterations, stopping")
                    break
                
                # Check termination
                if self._check_termination():
                    break
                    
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                self.running = False
            except Exception as e:
                logger.exception(f"Error in iteration {self.iteration}")
        
        # Final test
        solution = self._build_solution()
        solution = self._final_test(solution)
        
        return solution
    
    def _init_session(self, goal: str, initial_code: str):
        """Initialize search session."""
        self.session_id = str(uuid.uuid4())[:8]
        self.iteration = 0
        self.score_history = []
        self.retry_count = 0
        
        logger.info(f"Initializing session {self.session_id}")
        
        # Create Git branch
        branch_name = f"{self.session_id}/root"
        try:
            self.git_manager.create_branch(branch_name)
        except:
            pass
        
        # Write initial code
        if not initial_code:
            initial_code = f"# Goal: {goal[:50]}\n\npass\n"
        
        self.git_manager.write_file("program.py", initial_code)
        initial_commit = self.git_manager.snapshot(f"MCTS: initial - {goal[:30]}")
        
        # Create search tree
        self.search_tree = SearchTree(
            session_id=self.session_id,
            goal=goal,
            num_islands=self.config.num_islands,
            exploration_weight=self.config.exploration_weight,
            migration_interval=self.config.migration_interval,
            migration_rate=self.config.migration_rate,
        )
        
        # Create root node
        root_node = MCTSNode(
            id=str(uuid.uuid4())[:8],
            commit_hash=initial_commit,
            session_id=self.session_id,
            depth=0,
            island_id=0,
        )
        root_node.code = self.git_manager.get_code(initial_commit, "program.py") or initial_code
        
        # Light evaluate root
        self._light_evaluate(root_node)
        
        # Add to tree
        self.search_tree.root_id = root_node.id
        self.search_tree.add_node(root_node)
        self.search_tree.init_islands(root_node.id)
        
        logger.info(f"Session initialized: {goal}")
    
    def _run_iteration(self):
        """Run a single MCTS iteration with 2-branch optimization."""
        # Select node to expand
        node = self._select_node()
        
        if node is None:
            return
        
        if node.depth >= self.config.max_depth:
            return
        
        # Expand: generate 2 actions
        actions = self._expand(node)
        
        if not actions:
            return
        
        # Sort by confidence (higher first)
        actions.sort(key=lambda a: a.confidence if hasattr(a, 'confidence') and a.confidence else 0.5, reverse=True)
        
        # Execute actions (try best first, fallback to second if failed)
        executed_any = False
        
        for i, action in enumerate(actions):
            if i > 0 and executed_any:
                # Already have a successful child, skip remaining
                break
            
            child = self._try_action(node, action)
            
            if child is not None:
                self._light_evaluate(child)
                self._backpropagate(child)
                self._update_feature_grid(child)
                executed_any = True
                
                # If this was the top choice and succeeded, don't try others
                if i == 0 and action.confidence and action.confidence > 0.7:
                    break
    
    def _select_node(self) -> Optional[MCTSNode]:
        """Select node to expand using UCB."""
        island = self.search_tree.get_next_island()
        current = self.search_tree.get_node(island.root_id)
        
        if current is None:
            return self.search_tree.get_root()
        
        # Traverse down using UCB
        while True:
            children = self.search_tree.get_children(current.id)
            valid_children = [c for c in children if c.status != NodeStatus.PRUNED]
            
            if not valid_children:
                return current
            
            best_child = None
            best_ucb = float('-inf')
            
            for child in valid_children:
                ucb = self._calculate_ucb(child, current)
                child.ucb_score = ucb
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            
            if best_child is None:
                return current
            
            current = best_child
    
    def _calculate_ucb(self, node: MCTSNode, parent: MCTSNode) -> float:
        """Calculate UCB score."""
        if node.visits == 0:
            return float('inf')
        
        c = self.config.exploration_weight * (self.config.exploration_decay ** self.iteration)
        
        exploitation = node.value_avg
        exploration = c * math.sqrt(math.log(max(parent.visits, 1)) / node.visits)
        
        return exploitation + exploration
    
    def _expand(self, node: MCTSNode) -> List[Action]:
        """
        Expand node by generating actions.
        
        Strategy:
        1. First iteration: always use 2 branches (exploration phase)
        2. Subsequent: use self-assessment confidence to decide branches
        
        Self-assessment is more accurate than token entropy for task-level uncertainty.
        """
        if not self.llm_client:
            return self._generate_simple_actions(node)
        
        # Determine number of branches
        num_branches = 2  # Default
        
        # For iterations > 1, use self-assessment to decide
        if self.iteration > 0 and node.parent_id:  # Not root
            # Ask model to self-rate confidence for THIS specific goal
            assess_prompt = f"""Rate your confidence (0.0-1.0) for solving this programming task:

Goal: {self.search_tree.goal}

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
                assess_response = asyncio.run(
                    self.llm_client.generate(
                        prompt=assess_prompt,
                        temperature=0.1,
                        max_tokens=10,
                    )
                )
                
                # Parse confidence
                import re
                match = re.search(r'(\d+\.?\d*)', assess_response.content)
                if match:
                    self_confidence = float(match.group(1))
                    
                    # Use entropy as additional signal
                    entropy = assess_response.entropy
                    
                    logger.info(f"Self-assessment: confidence={self_confidence:.3f}, entropy={entropy:.3f}")
                    
                    # Decision logic:
                    # High self-confidence (>0.8) + Low entropy (<0.05) → single branch
                    # Otherwise → multi branch
                    if self_confidence > 0.8 and entropy < 0.05:
                        num_branches = 1
                        logger.info(f"High confidence detected, using single branch")
                    else:
                        num_branches = 2
                        
            except Exception as e:
                logger.warning(f"Self-assessment failed: {e}")
                num_branches = 2  # Default to multi
        
        # Generate actions
        inspirations = self._get_inspirations(node)
        previous_attempts = self._get_previous_attempts(node)
        
        prompt = self.prompt_builder.build_expand_prompt(
            goal=self.search_tree.goal,
            current_code=node.code,
            node=node,
            inspirations=inspirations,
            previous_attempts=previous_attempts,
            artifacts=node.artifacts,
            num_actions=num_branches,
        )
        
        try:
            response = asyncio.run(
                self.llm_client.generate_json(
                    prompt=prompt["user"],
                    system=prompt["system"],
                    temperature=0.7,
                )
            )
            
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
            
            # Pad if needed
            if len(actions) < num_branches:
                simple_actions = self._generate_simple_actions(node)
                actions.extend(simple_actions[:num_branches - len(actions)])
            
            return actions[:num_branches]
            
        except Exception as e:
            logger.warning(f"LLM expansion failed: {e}")
            return self._generate_simple_actions(node)[:num_branches]
    
    def _generate_simple_actions(self, node: MCTSNode) -> List[Action]:
        """Generate simple actions without LLM."""
        goal = self.search_tree.goal.lower()
        actions = []
        
        # Default confidence for simple actions
        default_confidence = 0.6
        
        if "hello" in goal or "world" in goal:
            actions.append(Action(
                id=str(uuid.uuid4())[:8],
                description="Add hello world function",
                reasoning="Simple hello world implementation",
                confidence=0.8,
                tool_calls=[{
                    "tool": "write",
                    "args": {
                        "path": "program.py",
                        "content": "def hello_world():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    hello_world()\n"
                    }
                }]
            ))
        elif "class" in goal:
            import re
            class_match = re.search(r'class\s+(?:called\s+)?(\w+)', self.search_tree.goal)
            class_name = class_match.group(1) if class_match else "MyClass"
            
            actions.append(Action(
                id=str(uuid.uuid4())[:8],
                description=f"Add {class_name} class",
                reasoning="Basic class structure",
                confidence=default_confidence,
                tool_calls=[{
                    "tool": "write",
                    "args": {
                        "path": "program.py",
                        "content": f"class {class_name}:\n    def __init__(self):\n        pass\n"
                    }
                }]
            ))
        else:
            actions.append(Action(
                id=str(uuid.uuid4())[:8],
                description="Improve code",
                reasoning="Generic improvement",
                confidence=default_confidence,
                tool_calls=[{
                    "tool": "write",
                    "args": {
                        "path": "program.py",
                        "content": f"# Implementation for: {self.search_tree.goal[:50]}\npass\n"
                    }
                }]
            ))
        
        # Add second action as variation
        actions.append(Action(
            id=str(uuid.uuid4())[:8],
            description="Add documentation",
            reasoning="Improve code documentation",
            confidence=0.4,
            tool_calls=[{
                "tool": "edit",
                "args": {
                    "path": "program.py",
                    "old": "def ",
                    "new": '"""Function docstring."""\ndef '
                }
            }]
        ))
        
        return actions[:self.config.num_actions_per_expand]
    
    def _get_inspirations(self, node: MCTSNode) -> List[MCTSNode]:
        """Get inspiration nodes."""
        inspirations = []
        
        exclude_coords = None
        if node.feature_coords:
            exclude_coords = node.feature_coords.to_key()
        
        diverse_ids = self.search_tree.feature_grid.get_diverse_nodes(
            exclude_coords=exclude_coords,
            n=self.config.num_inspirations // 2,
        )
        
        for nid in diverse_ids:
            n = self.search_tree.get_node(nid)
            if n:
                inspirations.append(n)
        
        if self.search_tree.best_node_id:
            best = self.search_tree.get_node(self.search_tree.best_node_id)
            if best and best.id != node.id:
                inspirations.append(best)
        
        return inspirations[:self.config.num_inspirations]
    
    def _get_previous_attempts(self, node: MCTSNode) -> List[Dict]:
        """Get previous attempts from path to root."""
        attempts = []
        path = self.search_tree.get_path_to_root(node.id)
        
        for n in path[1:4]:
            if n.action:
                attempts.append({
                    "description": n.action.description,
                    "error": n.action.error,
                    "score": n.value_avg,
                })
        
        return attempts
    
    def _try_action(self, parent: MCTSNode, action: Action) -> Optional[MCTSNode]:
        """Try to execute an action."""
        self.git_manager.restore(parent.commit_hash)
        
        for tool_call in action.tool_calls:
            result = self.tool_executor.execute(tool_call)
            
            if not result.success:
                action.success = False
                action.error = result.error
                self.git_manager.restore(parent.commit_hash)
                return None
        
        action.success = True
        commit_hash = self.git_manager.snapshot(f"MCTS: {action.description[:40]}")
        
        child = MCTSNode(
            id=str(uuid.uuid4())[:8],
            commit_hash=commit_hash,
            session_id=self.session_id,
            parent_id=parent.id,
            depth=parent.depth + 1,
            action=action,
            island_id=parent.island_id,
        )
        
        child.code = self.git_manager.get_code(commit_hash, "program.py") or ""
        parent.children_ids.append(child.id)
        self.search_tree.add_node(child)
        
        return child
    
    def _light_evaluate(self, node: MCTSNode):
        """
        Light evaluation during MCTS search.
        
        Fast checks without heavy LLM calls.
        """
        result = EvaluationResult()
        
        # 1. Syntax check (local, fast)
        syntax_ok = self._check_syntax(node.code)
        result.metrics["syntax"] = 1.0 if syntax_ok else 0.0
        
        if not syntax_ok:
            result.score = 0.0
            node.evaluation = result
            node.status = NodeStatus.EVALUATED
            node.update_stats(0.0)
            return
        
        # 2. Check if code is meaningful (not just pass/comments)
        code_quality = self._quick_code_quality(node.code)
        result.metrics["quality"] = code_quality
        
        # 3. Quick goal relevance (keyword matching)
        relevance = self._quick_relevance(node.code, self.search_tree.goal)
        result.metrics["relevance"] = relevance
        
        # 4. Quick LLM check (only if code looks promising)
        if code_quality > 0.3 and relevance > 0.3 and self.llm_client:
            llm_score = self._quick_llm_check(node.code, self.search_tree.goal)
            result.metrics["llm_quick"] = llm_score
        else:
            result.metrics["llm_quick"] = 0.3
        
        # Calculate score
        result.score = (
            0.2 * result.metrics["syntax"] +
            0.3 * result.metrics["quality"] +
            0.2 * result.metrics["relevance"] +
            0.3 * result.metrics["llm_quick"]
        )
        
        node.evaluation = result
        node.status = NodeStatus.EVALUATED
        node.update_stats(result.score)
    
    def _check_syntax(self, code: str) -> bool:
        """Check if code has valid syntax."""
        import ast
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _quick_code_quality(self, code: str) -> float:
        """Quick code quality check (local)."""
        if not code or len(code.strip()) < 10:
            return 0.0
        
        # Check for placeholder code
        if code.strip().endswith("pass") and len(code.strip()) < 50:
            return 0.1
        
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
        
        # Check for proper indentation (multiple lines with consistent indent)
        lines = code.split("\n")
        indented_lines = [l for l in lines if l.startswith("    ") or l.startswith("\t")]
        if len(indented_lines) > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _quick_relevance(self, code: str, goal: str) -> float:
        """Quick relevance check using keywords."""
        goal_lower = goal.lower()
        code_lower = code.lower()
        
        # Important: we want to check if the code implements the core concept
        # not whether it contains the exact words from the goal
        
        # Core programming concepts that indicate relevance
        concept_keywords = {
            "prime": ["prime", "is_prime", "check_prime"],
            "sort": ["sort", "sorted", "quicksort", "mergesort", "bubblesort"],
            "search": ["search", "find", "binary_search", "linear_search"],
            "tree": ["tree", "node", "binary_tree", "bst"],
            "list": ["list", "array", "linked_list"],
            "stack": ["stack", "push", "pop"],
            "queue": ["queue", "enqueue", "dequeue"],
            "graph": ["graph", "vertex", "edge", "bfs", "dfs"],
            "palindrome": ["palindrome", "is_palindrome"],
            "factorial": ["factorial", "fact"],
            "fibonacci": ["fibonacci", "fib"],
            "reverse": ["reverse", "reversed"],
            "binary": ["binary", "0b", "bin("],
            "hash": ["hash", "dict", "dictionary", "map"],
        }
        
        # Find what concept the goal is about
        matched_concepts = []
        for concept, keywords in concept_keywords.items():
            if concept in goal_lower:
                matched_concepts.append((concept, keywords))
        
        if not matched_concepts:
            # No specific concept found, check for general programming terms
            if "def " in code_lower or "class " in code_lower:
                return 0.6
            return 0.3
        
        # Check if code mentions any of the concept keywords
        total_score = 0
        for concept, keywords in matched_concepts:
            for kw in keywords:
                if kw in code_lower:
                    total_score += 1
                    break  # Only count each concept once
        
        return min(total_score / len(matched_concepts), 1.0)
    
    def _quick_llm_check(self, code: str, goal: str) -> float:
        """Quick LLM check - simplified prompt for speed."""
        if not self.llm_client:
            return 0.5
        
        try:
            prompt = f"""Rate this code for the goal (0.0-1.0). Reply with just the number.

Goal: {goal[:100]}

Code:
```
{code[:500]}
```

Score:"""

            response = asyncio.run(
                self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=10,
                )
            )
            
            # Extract number from response
            import re
            match = re.search(r'(\d+\.?\d*)', response.content)
            if match:
                return float(match.group(1))
            return 0.5
            
        except Exception as e:
            logger.debug(f"Quick LLM check failed: {e}")
            return 0.5
    
    def _backpropagate(self, node: MCTSNode):
        """Backpropagate value up the tree."""
        value = node.evaluation.score if node.evaluation else 0.0
        
        path = self.search_tree.get_path_to_root(node.id)
        
        for n in path:
            n.update_stats(value)
        
        if node.island_id < len(self.search_tree.islands):
            island = self.search_tree.islands[node.island_id]
            island.total_visits += 1
            island.update_best(node)
        
        if node.value_avg > self.search_tree.best_value:
            self.search_tree.best_value = node.value_avg
            self.search_tree.best_node_id = node.id
        
        self.score_history.append(self.search_tree.best_value)
    
    def _update_feature_grid(self, node: MCTSNode):
        """Update feature grid with node."""
        coords = self._calculate_feature_coords(node)
        node.feature_coords = coords
        
        key = coords.to_key()
        existing_id = self.search_tree.feature_grid.grid.get(key)
        
        if existing_id is None:
            self.search_tree.feature_grid.grid[key] = node.id
            self.search_tree.feature_grid.node_to_coords[node.id] = key
        else:
            existing = self.search_tree.get_node(existing_id)
            if existing and node.value_avg > existing.value_avg:
                if existing_id in self.search_tree.feature_grid.node_to_coords:
                    del self.search_tree.feature_grid.node_to_coords[existing_id]
                self.search_tree.feature_grid.grid[key] = node.id
                self.search_tree.feature_grid.node_to_coords[node.id] = key
    
    def _calculate_feature_coords(self, node: MCTSNode) -> FeatureCoords:
        """Calculate MAP-Elites feature coordinates."""
        code = node.code
        
        lines = len(code.split("\n"))
        complexity = min(9, lines // 30)
        
        code_lower = code.lower()
        if "class " in code_lower:
            approach = 0
        elif "def " in code_lower:
            approach = 1
        elif "import " in code_lower:
            approach = 2
        else:
            approach = 3
        
        quality_tier = min(9, int(node.value_avg * 10))
        
        return FeatureCoords(
            complexity=complexity,
            approach=approach,
            quality_tier=quality_tier,
        )
    
    def _check_termination(self) -> bool:
        """Check if search should terminate."""
        if self.search_tree.best_value >= self.config.target_score:
            logger.info(f"Target score reached: {self.search_tree.best_value:.3f}")
            return True
        
        if len(self.search_tree.nodes) >= self.config.max_nodes:
            logger.info("Max nodes reached")
            return True
        
        return False
    
    def _final_test(self, solution: Solution) -> Solution:
        """
        Final testing of the solution.
        
        Runs tests and LLM code review.
        """
        if not solution.best_node:
            return solution
        
        logger.info("Running final test...")
        
        code = solution.best_code
        goal = solution.goal
        
        # 1. Run tests if available
        test_result = self._run_tests(code)
        solution.test_passed = test_result.get("passed", False)
        solution.test_output = test_result.get("output", "")
        
        # 2. LLM code review
        if self.llm_client:
            review = self._llm_code_review(code, goal)
            solution.best_score = review.get("score", solution.best_score)
            
            # If test failed but review is good, note the issue
            if not solution.test_passed and solution.best_score > 0.7:
                logger.warning("Code review passed but tests failed")
        
        # 3. If failed and can retry, do retry
        if not solution.test_passed and self.retry_count < self.config.max_retries:
            self.retry_count += 1
            logger.info(f"Test failed, retrying ({self.retry_count}/{self.config.max_retries})")
            
            # Continue MCTS for more iterations
            for _ in range(3):
                self._run_iteration()
                self.iteration += 1
            
            # Rebuild solution
            solution = self._build_solution()
            return self._final_test(solution)
        
        return solution
    
    def _run_tests(self, code: str) -> Dict[str, Any]:
        """Run tests on the code."""
        result = {"passed": True, "output": ""}
        
        # Check for test files
        test_files = self._find_test_files()
        
        if not test_files:
            # No tests, check if code at least runs
            try:
                import subprocess
                proc = subprocess.run(
                    ["python", "-c", f"import ast; ast.parse('''{code}''')"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode != 0:
                    result["passed"] = False
                    result["output"] = f"Syntax error: {proc.stderr}"
            except Exception as e:
                result["output"] = str(e)
            
            return result
        
        # Run pytest
        try:
            import subprocess
            proc = subprocess.run(
                ["python", "-m", "pytest"] + test_files + ["-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            result["output"] = proc.stdout + proc.stderr
            result["passed"] = proc.returncode == 0
            
        except Exception as e:
            result["output"] = str(e)
            result["passed"] = False
        
        return result
    
    def _find_test_files(self) -> List[str]:
        """Find test files."""
        test_files = []
        
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if not d.startswith((".", "_", "node_modules", "venv"))]
            
            for f in files:
                if f.startswith("test_") and f.endswith(".py"):
                    test_files.append(os.path.join(root, f))
        
        return test_files
    
    def _llm_code_review(self, code: str, goal: str) -> Dict[str, Any]:
        """LLM code review for final evaluation."""
        if not self.llm_client:
            return {"score": 0.5}
        
        try:
            prompt = f"""Review this code for the given goal.

Goal: {goal}

Code:
```
{code[:2000]}
```

Provide a JSON review:
{{
  "score": 0.0-1.0,
  "correctness": "does it solve the goal?",
  "issues": ["issue1", "issue2"],
  "strengths": ["strength1"],
  "suggestions": ["suggestion1"]
}}"""

            response = asyncio.run(
                self.llm_client.generate_json(
                    prompt=prompt,
                    system="You are a code reviewer. Provide constructive feedback.",
                    temperature=0.3,
                )
            )
            
            return response
            
        except Exception as e:
            logger.warning(f"LLM code review failed: {e}")
            return {"score": 0.5}
    
    def _build_solution(self) -> Solution:
        """Build final solution."""
        best_node = None
        if self.search_tree.best_node_id:
            best_node = self.search_tree.get_node(self.search_tree.best_node_id)
        
        best_code = best_node.code if best_node else ""
        
        if self.config.auto_merge_best and best_node:
            try:
                self.git_manager.merge_to_main(
                    best_node.commit_hash,
                    f"MCTS: best solution ({self.search_tree.best_value:.3f})"
                )
            except Exception as e:
                logger.warning(f"Failed to merge: {e}")
        
        return Solution(
            session_id=self.session_id,
            goal=self.search_tree.goal,
            best_code=best_code,
            best_score=self.search_tree.best_value,
            best_node=best_node,
            search_tree=self.search_tree,
            iterations=self.iteration,
            total_nodes=len(self.search_tree.nodes),
        )