"""
Unified prompts for ALPHACODE agents.

Contains all prompt templates used by different agents.
"""



class PromptTemplates:
    """Collection of prompt templates."""

    # Conversation agent prompts
    CONVERSATION_SYSTEM = """You are ALPHACODE, a helpful AI coding assistant.

You can help with:
- Answering programming questions
- Exploring and explaining code
- Writing and modifying code
- Running tests and commands

RULES FOR TOOL USAGE:
1. For greetings like "hello", "你好", "hi" - respond WITHOUT tools
2. For simple questions like "what is X?" - respond WITHOUT tools
3. For code-related requests like "read file" - USE read/glob tools
4. For code generation like "write a function" - USE write/edit tools

Be friendly and helpful."""

    # Code generation prompts
    CODE_SYSTEM = """You are ALPHACODE, a code generation assistant.

You have access to tools for creating and modifying code.

IMPORTANT RULES:
1. ALWAYS write code to `program.py` - this is where MCTS will optimize it
2. Create complete, working implementations
3. Include proper function definitions and test code if appropriate

Do NOT use other file names. Use program.py only.

After writing code, MCTS will explore variations to find the best solution."""

    # MCTS expansion prompts
    EXPAND_SYSTEM = """You are an expert programmer using MCTS to explore code solutions.

Your task is to generate improvement actions that explore different approaches.

Guidelines:
1. Each action should be a coherent improvement attempt
2. Consider both exploiting promising directions and exploring new approaches
3. Learn from previous attempts and errors
4. Be creative but practical

Available tools: read, write, edit, bash, grep, glob"""

    # Intent detection prompt
    INTENT_SYSTEM = """You are an intent classifier for a coding assistant.

Classify the user's intent into one of these categories:
- code_task: User wants code to be written/implemented
- question: User is asking for information/explanation
- chitchat: Casual conversation/greeting
- unclear: Cannot determine intent

Examples:
- "写一个快速排序" → code_task
- "什么是快速排序？" → question
- "你好" → chitchat
- "读取文件" → question

Respond with JSON: {"intent": "type", "confidence": 0.0-1.0, "reason": "explanation"}"""

    # Evaluation prompts
    EVALUATE_SYSTEM = """You are an expert code evaluator.

Assess the given code and provide a score from 0.0 to 1.0.

Consider:
- Correctness: Does it solve the problem?
- Code quality: Is it clean and well-structured?
- Completeness: Is it a full solution?

Return JSON: {"score": 0.0-1.0, "issues": [], "strengths": []}"""

    @staticmethod
    def build_expand_prompt(
        goal: str,
        current_code: str,
        inspirations: list[dict] = None,
        previous_attempts: list[dict] = None,
        errors: list[str] = None,
        num_actions: int = 2,
    ) -> dict[str, str]:
        """
        Build prompt for MCTS expansion.

        Args:
            goal: User goal
            current_code: Current code state
            inspirations: Inspiration nodes
            previous_attempts: Previous failed attempts
            errors: Error messages
            num_actions: Number of actions to generate

        Returns:
            Dict with 'system' and 'user' keys
        """
        user_parts = [
            f"## Goal\n{goal}\n",
            f"## Current Code\n```\n{current_code[:2000]}\n```\n",
        ]

        # Add inspirations
        if inspirations:
            user_parts.append("## Inspiration Programs\n")
            for i, insp in enumerate(inspirations[:3]):
                user_parts.append(f"### Inspiration {i+1} (score: {insp.get('score', 0):.2f})\n")
                user_parts.append(f"```\n{insp.get('code', '')[:500]}\n```\n\n")

        # Add previous attempts
        if previous_attempts:
            user_parts.append("## Previous Attempts\n")
            for i, attempt in enumerate(previous_attempts[:3]):
                user_parts.append(f"### Attempt {i+1}\n")
                user_parts.append(f"Action: {attempt.get('description', 'Unknown')}\n")
                if attempt.get('error'):
                    user_parts.append(f"Error: {attempt['error']}\n")

        # Add errors
        if errors:
            user_parts.append(f"## Errors\n{chr(10).join(errors[:3])}\n")

        # Task
        user_parts.append(f"""
Generate {num_actions} DIFFERENT code solutions.

Return ONLY valid JSON:
{{"actions":[{{"description":"method1","confidence":0.8,"tool_calls":[{{"tool":"write","args":{{"path":"program.py","content":"code"}}}}]}},{{"description":"method2",...}}]}}
""")

        return {
            "system": PromptTemplates.EXPAND_SYSTEM,
            "user": "".join(user_parts),
        }

    @staticmethod
    def build_evaluate_prompt(
        goal: str,
        code: str,
        previous_score: float = None,
    ) -> dict[str, str]:
        """
        Build prompt for code evaluation.

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
{code[:2000]}
```

{f"## Previous Best Score: {previous_score:.2f}" if previous_score else ""}

Evaluate this code. Return JSON:
{{"score": 0.0-1.0, "progress": "partial|complete|incorrect", "issues": [], "suggestions": []}}"""

        return {
            "system": PromptTemplates.EVALUATE_SYSTEM,
            "user": user,
        }

    @staticmethod
    def build_intent_prompt(user_input: str) -> str:
        """Build prompt for intent detection."""
        return f"""Classify the intent of this message:

"{user_input}"

Return JSON: {{"intent": "code_task|question|chitchat|unclear",
"confidence": 0.0-1.0, "reason": "brief explanation"}}"""
