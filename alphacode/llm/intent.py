"""
Intent detection and conversation handling for ALPHACODE.

Classifies user input and responds appropriately to non-code intents.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """User intent types."""
    CODE_TASK = "code_task"        # 需要生成代码
    QUESTION = "question"          # 编程相关问题
    CHITCHAT = "chitchat"          # 闲聊
    UNCLEAR = "unclear"            # 不明确


@dataclass
class IntentResult:
    """Intent detection result."""
    intent: IntentType
    confidence: float
    reason: str
    code_hint: str | None = None  # 如果是代码任务，提取的具体需求


class IntentDetector:
    """
    Intent detector.

    Uses LLM to classify user input.
    """

    SYSTEM_PROMPT = """You are an intent classifier. Analyze the user's message and determine their intent.

Intent types:
- code_task: User wants code to be written, generated, or implemented
- question: User is asking a programming question (no code generation needed)
- chitchat: Casual conversation, greeting, or non-programming talk
- unclear: Cannot determine the intent clearly

Important: If intent is code_task, preserve the user's EXACT original request in code_hint, do not rewrite or summarize it.

Respond with JSON only: {"intent": "type", "confidence": 0.0-1.0, "reason": "brief explanation", "code_hint": "exact original request if code_task"}"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def detect(self, user_input: str) -> IntentResult:
        """
        Detect user intent.

        Args:
            user_input: User's input message

        Returns:
            IntentResult with classification
        """
        if not self.llm_client:
            # No LLM, assume it's a code task
            return IntentResult(
                intent=IntentType.CODE_TASK,
                confidence=0.5,
                reason="No LLM available, assuming code task",
                code_hint=user_input,
            )

        prompt = f"""User message: "{user_input}"

Classify the intent. Is this asking for code to be written?"""

        try:
            response = await self.llm_client.generate_json(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                temperature=0.1,
            )

            intent_str = response.get("intent", "unclear")
            confidence = response.get("confidence", 0.5)
            reason = response.get("reason", "")
            code_hint = response.get("code_hint")

            # Map string to enum
            intent_map = {
                "code_task": IntentType.CODE_TASK,
                "question": IntentType.QUESTION,
                "chitchat": IntentType.CHITCHAT,
                "unclear": IntentType.UNCLEAR,
            }

            intent = intent_map.get(intent_str, IntentType.UNCLEAR)

            return IntentResult(
                intent=intent,
                confidence=confidence,
                reason=reason,
                code_hint=code_hint if intent == IntentType.CODE_TASK else None,
            )

        except Exception as e:
            logger.warning(f"Intent detection failed: {e}")
            # Default to code task to avoid missing actual tasks
            return IntentResult(
                intent=IntentType.CODE_TASK,
                confidence=0.5,
                reason="Detection failed, defaulting to code task",
                code_hint=user_input,
            )

    def detect_sync(self, user_input: str) -> IntentResult:
        """Synchronous intent detection."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.detect(user_input))


class ConversationHandler:
    """
    Handles non-code conversations.

    Actually responds to user questions and chitchat using LLM.
    """

    # System prompts for different intent types
    QUESTION_SYSTEM = """You are a helpful programming assistant. You are part of ALPHACODE, a code exploration tool.

When users ask programming questions:
1. Provide clear, accurate explanations
2. Include code examples when helpful
3. Be concise but thorough
4. If the user might want actual code implementation, suggest they ask ALPHACODE to generate it

Remember: You can explain concepts, but for actual code generation tasks, the main MCTS exploration engine will handle that."""

    CHITCHAT_SYSTEM = """You are a friendly AI assistant named ALPHACODE. You are a code exploration tool that helps users write and improve code through iterative MCTS-based search.

Be conversational and helpful. If the user seems interested in coding, encourage them to describe what they'd like to build.

Keep responses brief and friendly."""

    UNCLEAR_SYSTEM = """You are ALPHACODE, a code exploration assistant.

If you're not sure what the user wants, politely ask for clarification. Mention that you specialize in:
- Writing new code
- Improving existing code
- Exploring different implementation approaches

Guide the user toward describing a coding task if that's what they need."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.conversation_history: list[dict] = []

    async def respond(
        self,
        user_input: str,
        intent_result: IntentResult,
    ) -> str:
        """
        Generate a real response for non-code intents.

        Args:
            user_input: User's original input
            intent_result: Detected intent

        Returns:
            LLM-generated response
        """
        if not self.llm_client:
            # Fallback to static responses if no LLM
            return self._fallback_response(intent_result)

        # Select appropriate system prompt
        system_prompt = self._get_system_prompt(intent_result.intent)

        # Build context with conversation history
        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": user_input})

        try:
            response = await self.llm_client.generate_with_context(
                system=system_prompt,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})

            # Limit history length
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            return response

        except Exception as e:
            logger.error(f"Conversation response failed: {e}")
            return self._fallback_response(intent_result)

    def _get_system_prompt(self, intent: IntentType) -> str:
        """Get appropriate system prompt for intent type."""
        if intent == IntentType.QUESTION:
            return self.QUESTION_SYSTEM
        elif intent == IntentType.CHITCHAT:
            return self.CHITCHAT_SYSTEM
        else:
            return self.UNCLEAR_SYSTEM

    def _fallback_response(self, intent_result: IntentResult) -> str:
        """Fallback static response when LLM is unavailable."""
        if intent_result.intent == IntentType.QUESTION:
            return (
                "I'd love to answer your question, but my LLM connection isn't working. "
                "Please check the API configuration. "
                "Once configured, I can explain programming concepts and help write code!"
            )
        elif intent_result.intent == IntentType.CHITCHAT:
            return (
                "Hello! I'm ALPHACODE, a code exploration assistant. "
                "I can help you write and improve code through iterative search. "
                "Tell me what code you'd like me to create!"
            )
        else:
            return (
                "I'm not sure what you're asking for. "
                "I specialize in writing and improving code. "
                "Could you describe what code you'd like me to create?"
            )

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def respond_sync(self, user_input: str, intent_result: IntentResult) -> str:
        """Synchronous response generation."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.respond(user_input, intent_result))


# Keep for backwards compatibility, but prefer ConversationHandler
def get_response_for_intent(intent_result: IntentResult) -> str:
    """
    Legacy function - returns static responses.
    Use ConversationHandler.respond() for actual LLM responses.
    """
    if intent_result.intent == IntentType.QUESTION:
        return (
            "I'm designed to help write code through iterative exploration. "
            "For programming questions, I can try to generate example code if you'd like. "
            "Would you like me to create some code related to your question?"
        )

    elif intent_result.intent == IntentType.CHITCHAT:
        return (
            "Hello! I'm ALPHACODE, a code exploration assistant. "
            "I can help you write and improve code through iterative search. "
            "Tell me what code you'd like me to create!"
        )

    elif intent_result.intent == IntentType.UNCLEAR:
        return (
            "I'm not sure what you're asking for. "
            "I specialize in writing and improving code. "
            "Could you describe what code you'd like me to create?"
        )

    return ""
