"""
Intent detection for MCTS-Agent.

Classifies user input to determine if it's a programming task.
"""

import logging
from typing import Dict, Any, Optional
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
    code_hint: Optional[str] = None  # 如果是代码任务，提取的具体需求


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
                reason=f"Detection failed, defaulting to code task",
                code_hint=user_input,
            )
    
    def detect_sync(self, user_input: str) -> IntentResult:
        """Synchronous intent detection."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.detect(user_input))


def get_response_for_intent(intent_result: IntentResult) -> str:
    """Generate appropriate response for non-code intents."""
    if intent_result.intent == IntentType.QUESTION:
        return (
            "I'm designed to help write code through iterative exploration. "
            "For programming questions, I can try to generate example code if you'd like. "
            "Would you like me to create some code related to your question?"
        )
    
    elif intent_result.intent == IntentType.CHITCHAT:
        return (
            "Hello! I'm MCTS-Agent, a code exploration assistant. "
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