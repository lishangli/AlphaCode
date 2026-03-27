"""
Prompt builder for MCTS-Agent.

Constructs prompts for LLM interactions.
"""

from typing import Dict, List, Any, Optional
from alphacode.core.node import MCTSNode, EvaluationResult


class PromptBuilder:
    """
    Prompt builder.
    
    Constructs prompts for:
    - Action generation (expand phase)
    - Progress evaluation
    - Code quality assessment
    """
    
    # System prompts
    EXPAND_SYSTEM = """You are an expert programmer using MCTS (Monte Carlo Tree Search) to explore code solutions.

Your task is to generate improvement actions that explore different approaches to solve the given problem.

Guidelines:
1. Each action should be a coherent improvement attempt
2. Consider both exploiting promising directions and exploring new approaches
3. Learn from previous attempts and errors
4. Be creative but practical

Available tools:
- read(path, offset?, limit?): Read file contents
- write(path, content): Write to file
- edit(path, old, new, all?): Replace text in file
- bash(cmd, timeout?): Execute shell command
- grep(pattern, path?): Search for pattern
- glob(pattern, path?): Find files

Remember: You're building a search tree. Try to propose actions that explore diverse approaches."""

    EVALUATE_SYSTEM = """You are an expert code evaluator. Assess the given code and provide a score from 0.0 to 1.0.

Consider:
- Correctness: Does it solve the problem?
- Code quality: Is it clean, readable, and well-structured?
- Completeness: Is it a full solution or partial?

Return only a JSON object with your assessment."""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates."""
        return {
            "expand_user": self._expand_user_template,
            "evaluate_user": self._evaluate_user_template,
        }

    def _evaluate_user_template(self, **kwargs) -> str:
        """Template for evaluate user prompt."""
        # Placeholder for template system
        return ""
    
    def build_expand_prompt(
        self,
        goal: str,
        current_code: str,
        node: MCTSNode,
        inspirations: List[MCTSNode],
        previous_attempts: List[Dict],
        artifacts: Dict[str, Any],
        num_actions: int = 3,
    ) -> Dict[str, str]:
        """
        Build prompt for expand phase.
        
        Args:
            goal: User goal
            current_code: Current code state
            node: Current node
            inspirations: Inspiration nodes
            previous_attempts: Previous failed attempts
            artifacts: Error messages, outputs
            num_actions: Number of actions to generate
            
        Returns:
            Dict with 'system' and 'user' keys
        """
        user_parts = [
            f"## Goal\n{goal}\n",
            f"## Current Code\n```\n{self._truncate_code(current_code, 2000)}\n```\n",
        ]
        
        # Current metrics
        if node and node.evaluation:
            user_parts.append(
                f"## Current Metrics\n{self._format_metrics(node.evaluation)}\n"
            )
        
        # Errors from last attempt
        if artifacts and node:
            error_section = self._format_artifacts(artifacts)
            if error_section:
                user_parts.append(f"## Errors\n{error_section}\n")
        
        # Previous attempts
        if previous_attempts and node:
            user_parts.append("## Previous Attempts\n")
            for i, attempt in enumerate(previous_attempts[-3:]):
                user_parts.append(f"### Attempt {i+1}\n")
                user_parts.append(f"Action: {attempt.get('description', 'Unknown')}\n")
                if attempt.get('error'):
                    user_parts.append(f"Error: {attempt['error']}\n")
                if attempt.get('score'):
                    user_parts.append(f"Score: {attempt['score']:.2f}\n")
        
        # Inspirations
        if inspirations:
            user_parts.append("## Inspiration Programs\n")
            user_parts.append("These are successful solutions from different approaches:\n\n")
            for i, insp in enumerate(inspirations):
                user_parts.append(f"### Inspiration {i+1} (score: {insp.value_avg:.2f})\n")
                user_parts.append(f"```\n{self._truncate_code(insp.code, 500)}\n```\n\n")
        
        # Task
        user_parts.append(f"""
Generate 2 DIFFERENT code solutions for the goal above.

Return ONLY valid JSON (no markdown, no explanation):
{{"actions":[{{"description":"method1","confidence":0.8,"tool_calls":[{{"tool":"write","args":{{"path":"program.py","content":"code here"}}}}]}},{{"description":"method2","confidence":0.7,"tool_calls":[{{"tool":"write","args":{{"path":"program.py","content":"code here"}}}}]}}]}}
""")
        
        return {
            "system": self.EXPAND_SYSTEM,
            "user": "\n".join(user_parts),
        }
    
    def build_evaluate_prompt(
        self,
        goal: str,
        code: str,
        previous_score: float = None,
    ) -> Dict[str, str]:
        """
        Build prompt for progress evaluation.
        
        Args:
            goal: User goal
            code: Code to evaluate
            previous_score: Previous best score
            
        Returns:
            Dict with 'system' and 'user' keys
        """
        user = f"""## Goal
{goal}

## Code
```
{self._truncate_code(code, 2000)}
```

{f"## Previous Best Score: {previous_score:.2f}" if previous_score else ""}

Evaluate this code's progress toward the goal. Return JSON:
```json
{{
  "score": 0.0-1.0,
  "progress": "partial|complete|incorrect",
  "issues": ["issue1", "issue2"],
  "strengths": ["strength1"],
  "suggestions": ["suggestion1"]
}}
```"""
        
        return {
            "system": self.EVALUATE_SYSTEM,
            "user": user,
        }
    
    def build_quality_prompt(self, code: str) -> Dict[str, str]:
        """Build prompt for code quality assessment."""
        system = """You are a code quality expert. Assess the given code.
Return only a JSON object with your assessment."""
        
        user = f"""## Code
```
{self._truncate_code(code, 2000)}
```

Assess code quality. Return JSON:
```json
{{
  "score": 0.0-1.0,
  "readability": 0.0-1.0,
  "maintainability": 0.0-1.0,
  "efficiency": 0.0-1.0,
  "issues": ["issue1"],
  "suggestions": ["suggestion1"]
}}
```"""
        
        return {"system": system, "user": user}
    
    def _format_metrics(self, evaluation: EvaluationResult) -> str:
        """Format evaluation metrics."""
        lines = []
        for name, value in evaluation.metrics.items():
            if isinstance(value, float):
                lines.append(f"- {name}: {value:.3f}")
            else:
                lines.append(f"- {name}: {value}")
        lines.append(f"- **Score**: {evaluation.score:.3f}")
        return "\n".join(lines)
    
    def _format_artifacts(self, artifacts: Dict[str, Any]) -> str:
        """Format artifacts (errors, outputs)."""
        parts = []
        
        if artifacts.get("stderr"):
            parts.append(f"```\n{artifacts['stderr'][:500]}\n```")
        if artifacts.get("syntax_error"):
            parts.append(f"Syntax error: {artifacts['syntax_error']}")
        if artifacts.get("test_failures"):
            parts.append(f"Test failures:\n{artifacts['test_failures'][:500]}")
        
        return "\n".join(parts)
    
    def _truncate_code(self, code: str, max_length: int) -> str:
        """Truncate code if too long."""
        if len(code) <= max_length:
            return code
        return code[:max_length] + f"\n... (truncated, {len(code)} total chars)"
    
    def _expand_user_template(self, **kwargs) -> str:
        """Template for expand user prompt."""
        # Placeholder for template system
        return ""