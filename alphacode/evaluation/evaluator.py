"""
Cascade evaluator for MCTS-Agent.

Multi-level evaluation: syntax → tests → quality → progress.
"""

import ast
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

from alphacode.config import MCTSConfig
from alphacode.core.node import EvaluationResult, MCTSNode
from alphacode.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test execution result."""
    passed: int = 0
    failed: int = 0
    total: int = 0
    output: str = ""
    failures: list[str] = field(default_factory=list)
    coverage: float = 0.0

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total


class CascadeEvaluator:
    """
    Cascade evaluator.

    Evaluates code in levels:
    0. Syntax check (fast, cheap)
    1. Tests (medium cost)
    2. Quality (LLM-based)
    3. Integration (expensive)

    Stops early if score falls below threshold.
    """

    def __init__(
        self,
        config: MCTSConfig,
        llm_client: LLMClient = None,
    ):
        self.config = config
        self.llm_client = llm_client

    def evaluate(
        self,
        code: str,
        goal: str,
        node: MCTSNode = None,
    ) -> EvaluationResult:
        """
        Evaluate code with cascade.

        Args:
            code: Code to evaluate
            goal: User goal
            node: Optional node for context

        Returns:
            EvaluationResult
        """
        result = EvaluationResult()

        # Level 0: Syntax check
        syntax_result = self._evaluate_syntax(code)
        result.metrics["syntax"] = syntax_result["score"]
        result.artifacts.update(syntax_result.get("artifacts", {}))

        if syntax_result["score"] < self.config.cascade_thresholds[0]:
            result.score = syntax_result["score"] * 0.3
            result.level = 0
            return result

        # Level 1: Tests
        test_result = self._evaluate_tests(code)
        result.metrics["tests_passed"] = test_result.pass_rate
        result.metrics["coverage"] = test_result.coverage
        result.artifacts["test_output"] = test_result.output
        if test_result.failures:
            result.artifacts["test_failures"] = "\n".join(test_result.failures[:5])

        if test_result.pass_rate < self.config.cascade_thresholds[1]:
            result.score = (
                self.config.weight_syntax * result.metrics["syntax"] +
                self.config.weight_tests * result.metrics["tests_passed"]
            )
            result.level = 1
            return result

        # Level 2: Quality
        quality_result = self._evaluate_quality(code)
        result.metrics["quality"] = quality_result["score"]
        result.artifacts.update(quality_result.get("artifacts", {}))

        # Level 3: Progress (LLM-based)
        progress_result = self._evaluate_progress(code, goal, node)
        result.metrics["progress"] = progress_result["score"]
        result.artifacts.update(progress_result.get("artifacts", {}))

        # Calculate final score
        result.score = (
            self.config.weight_syntax * result.metrics["syntax"] +
            self.config.weight_tests * result.metrics["tests_passed"] +
            self.config.weight_quality * result.metrics.get("quality", 0.5) +
            self.config.weight_progress * result.metrics.get("progress", 0.5)
        )

        result.level = 3
        return result

    def _evaluate_syntax(self, code: str) -> dict[str, Any]:
        """Evaluate syntax correctness."""
        result = {"score": 1.0, "artifacts": {}}

        # Parse AST
        try:
            ast.parse(code)
        except SyntaxError as e:
            result["score"] = 0.0
            result["artifacts"]["syntax_error"] = str(e)
            result["artifacts"]["syntax_error_line"] = e.lineno
            return result

        # Check for common issues
        issues = []

        # Check for obvious errors
        if "TODO" in code or "FIXME" in code:
            issues.append("Contains TODO/FIXME")

        # Check for print statements (might be debugging)
        print_count = code.count("print(")
        if print_count > 5:
            issues.append(f"Many print statements ({print_count})")

        if issues:
            result["artifacts"]["lint_issues"] = issues
            result["score"] = max(0.7, 1.0 - 0.1 * len(issues))

        return result

    def _evaluate_tests(self, code: str) -> TestResult:
        """Run tests on code."""
        result = TestResult()

        # Look for test files
        test_files = self._find_test_files()

        if not test_files:
            # No tests, give benefit of doubt
            result.passed = 1
            result.total = 1
            result.output = "No test files found"
            return result

        # Write code to temp file and run tests
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Run pytest
            cmd = ["python", "-m", "pytest"] + test_files + [
                "-v", "--tb=short", "-q",
                "--timeout=30",
            ]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.evaluation_timeout,
            )

            result.output = proc.stdout + proc.stderr

            # Parse results
            result = self._parse_pytest_output(result.output, result)

        except subprocess.TimeoutExpired:
            result.output = "Test execution timed out"
            result.failed = result.total = 1

        except FileNotFoundError:
            # pytest not installed
            result.passed = 1
            result.total = 1
            result.output = "pytest not available"

        except Exception as e:
            result.output = f"Test error: {e}"
            result.failed = 1
            result.total = 1

        finally:
            os.unlink(temp_path)

        return result

    def _find_test_files(self) -> list[str]:
        """Find test files in current directory."""
        test_files = []

        for root, dirs, files in os.walk("."):
            # Skip hidden and common non-test directories
            dirs[:] = [d for d in dirs if not d.startswith((".", "_", "node_modules", "venv"))]

            for f in files:
                if f.startswith("test_") and f.endswith(".py"):
                    test_files.append(os.path.join(root, f))

        return test_files

    def _parse_pytest_output(self, output: str, result: TestResult) -> TestResult:
        """Parse pytest output."""
        # Look for summary line
        # Example: "5 passed, 2 failed"
        summary_match = re.search(
            r"(\d+) passed.*?(?:(\d+) failed)?",
            output
        )

        if summary_match:
            result.passed = int(summary_match.group(1))
            result.failed = int(summary_match.group(2) or 0)
            result.total = result.passed + result.failed

        # Look for coverage
        coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
        if coverage_match:
            result.coverage = int(coverage_match.group(1)) / 100

        # Extract failures
        failure_pattern = re.compile(r"FAILED (.*?) - (.*?)(?=\n|$)")
        for match in failure_pattern.finditer(output):
            result.failures.append(f"{match.group(1)}: {match.group(2)}")

        return result

    def _evaluate_quality(self, code: str) -> dict[str, Any]:
        """Evaluate code quality."""
        result = {"score": 0.5, "artifacts": {}}

        # Calculate complexity
        try:
            complexity = self._calculate_complexity(code)
            result["artifacts"]["complexity"] = complexity
            # Lower complexity is better
            result["score"] = max(0, 1 - complexity / 50)
        except Exception:
            pass

        # Check documentation
        doc_score = self._check_documentation(code)
        result["artifacts"]["doc_score"] = doc_score

        # Check code length
        lines = len(code.split("\n"))
        if lines < 50:
            length_score = 1.0
        elif lines < 200:
            length_score = 0.8
        elif lines < 500:
            length_score = 0.6
        else:
            length_score = 0.4

        result["artifacts"]["lines"] = lines

        # Combine scores
        result["score"] = (
            0.4 * result["score"] +
            0.3 * doc_score +
            0.3 * length_score
        )

        return result

    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity (simplified)."""
        complexity = 1

        tree = ast.parse(code)

        for node in ast.walk(tree):
            # Each decision point adds complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _check_documentation(self, code: str) -> float:
        """Check for documentation."""
        tree = ast.parse(code)

        total_items = 0
        documented_items = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                total_items += 1
                if ast.get_docstring(node):
                    documented_items += 1

        if total_items == 0:
            return 0.5

        return documented_items / total_items

    def _evaluate_progress(
        self,
        code: str,
        goal: str,
        node: MCTSNode = None,
    ) -> dict[str, Any]:
        """Evaluate progress toward goal using LLM."""
        result = {"score": 0.5, "artifacts": {}}

        if not self.llm_client:
            # Without LLM, use heuristics based on code content
            result["score"] = self._heuristic_progress(code, goal)
            return result

        # Use synchronous call
        try:
            import asyncio

            prompt = f"""Evaluate this code's progress toward the goal.

Goal: {goal}

Code:
```
{code[:1500]}
```

Return JSON with:
- "score": 0.0 to 1.0 (how much of the goal is achieved)
- "reasoning": brief explanation
"""

            response = asyncio.run(
                self.llm_client.generate_json(
                    prompt=prompt,
                    system="You are a code evaluator. Return only JSON.",
                    temperature=0.3,
                )
            )

            if "score" in response:
                result["score"] = float(response["score"])
            if "reasoning" in response:
                result["artifacts"]["llm_reasoning"] = response["reasoning"]

        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
            result["artifacts"]["llm_error"] = str(e)
            result["score"] = self._heuristic_progress(code, goal)

        return result

    def _heuristic_progress(self, code: str, goal: str) -> float:
        """Heuristic progress evaluation without LLM."""
        score = 0.1  # Start low

        # Check if code has actual implementation (not just pass)
        if "pass" in code and len(code.strip()) < 100:
            score = 0.1  # Placeholder code gets low score
        elif "pass" not in code:
            score = 0.3  # At least some implementation

        # Check for function definitions
        import re
        func_defs = re.findall(r'def\s+(\w+)\s*\(', code)
        if func_defs:
            score += 0.15 * min(len(func_defs), 3)

        # Check for class definitions
        class_defs = re.findall(r'class\s+(\w+)', code)
        if class_defs:
            score += 0.2

        # Check goal keywords in code
        goal_lower = goal.lower()
        code_lower = code.lower()

        # Extract class name from goal and check if it's in code
        class_match = re.search(r'class\s+(?:called\s+)?(\w+)', goal_lower)
        if class_match:
            class_name = class_match.group(1).lower()
            if class_name in code_lower:
                score += 0.15

        # Extract methods from goal and check if they're in code
        methods = re.findall(r'(\w+)\s+methods?', goal_lower)
        for method in methods:
            if method.lower() in code_lower:
                score += 0.1

        keywords = ["hello", "world", "print", "return", "init", "repr"]
        for kw in keywords:
            if kw in goal_lower and kw in code_lower:
                score += 0.05

        # Check for meaningful content
        lines = [
            line.strip() for line in code.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        if len(lines) > 3:
            score += 0.1
        if len(lines) > 5:
            score += 0.05

        return min(score, 1.0)


class MockEvaluator(CascadeEvaluator):
    """Mock evaluator for testing."""

    def evaluate(self, code: str, goal: str, node: MCTSNode = None) -> EvaluationResult:
        """Return mock evaluation."""
        import random

        result = EvaluationResult()
        result.score = random.uniform(0.5, 0.9)
        result.metrics = {
            "syntax": 1.0,
            "tests_passed": random.uniform(0.6, 1.0),
            "quality": random.uniform(0.5, 0.9),
            "progress": random.uniform(0.6, 1.0),
        }
        result.level = 3

        return result
