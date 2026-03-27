"""
LLM client for MCTS-Agent.

Handles communication with LLM APIs, including entropy-based confidence estimation.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM response."""
    content: str
    model: str
    usage: Dict[str, int]
    latency: float
    # Entropy information
    entropy: float = 0.5
    confidence: float = 0.5
    logprobs: Optional[List[float]] = None
    entropy_decision: str = "multi"  # "single" or "multi"


class LLMClient:
    """
    LLM client.
    
    Supports OpenAI-compatible APIs.
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 60,
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            api_base: API base URL
            model: Model name
            temperature: Generation temperature
            max_tokens: Maximum tokens
            timeout: Request timeout
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        if not self.api_key:
            logger.warning("No API key provided. Set OPENAI_API_KEY or pass api_key.")
    
    async def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
        logprobs: bool = False,  # Whether to return logprobs
        top_logprobs: int = 5,   # Number of top logprobs per token
    ) -> LLMResponse:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            system: System message
            temperature: Override temperature
            max_tokens: Override max tokens
            logprobs: Whether to return logprobs for entropy calculation
            top_logprobs: Number of top logprobs per token
            
        Returns:
            LLMResponse with optional entropy info
        """
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        return await self._call_api(
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
    
    async def generate_with_context(
        self,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        Generate with message context.
        
        Args:
            system: System message
            messages: Message history
            temperature: Override temperature
            max_tokens: Override max tokens
            
        Returns:
            Response content
        """
        full_messages = [{"role": "system", "content": system}]
        full_messages.extend(messages)
        
        response = await self._call_api(
            messages=full_messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        
        return response.content
    
    async def _call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> LLMResponse:
        """Make API call using httpx for better async support."""
        import httpx
        from alphacode.llm.entropy import EntropyAnalyzer
        
        start_time = time.time()
        
        request_data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add logprobs request
        if logprobs:
            request_data["logprobs"] = True
            request_data["top_logprobs"] = top_logprobs
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        url = f"{self.api_base}/chat/completions"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    json=request_data,
                    headers=headers,
                )
                
                if response.status_code != 200:
                    error_body = response.text
                    logger.error(f"API error: {response.status_code} - {error_body}")
                    raise Exception(f"API error: {response.status_code} - {error_body}")
                
                response_data = response.json()
                choice = response_data["choices"][0]
                content = choice["message"]["content"]
                
                # Calculate entropy if logprobs available
                entropy = 0.5
                confidence = 0.5
                token_logprobs = None
                entropy_decision = "multi"
                
                if logprobs and "logprobs" in choice and choice["logprobs"]:
                    logprobs_content = choice["logprobs"].get("content", [])
                    if logprobs_content:
                        # Extract logprobs
                        token_logprobs = [
                            t.get("logprob", 0) if isinstance(t, dict) else getattr(t, "logprob", 0)
                            for t in logprobs_content
                        ]
                        
                        # Analyze entropy
                        analyzer = EntropyAnalyzer()
                        entropy_result = analyzer.analyze(logprobs_content=logprobs_content)
                        
                        entropy = entropy_result.entropy
                        confidence = entropy_result.confidence
                        entropy_decision = entropy_result.decision
                        
                        logger.debug(f"Entropy analysis: {entropy_result}")
                
                return LLMResponse(
                    content=content,
                    model=response_data.get("model", self.model),
                    usage=response_data.get("usage", {}),
                    latency=time.time() - start_time,
                    entropy=entropy,
                    confidence=confidence,
                    logprobs=token_logprobs,
                    entropy_decision=entropy_decision,
                )
            
        except httpx.TimeoutException:
            logger.error(f"API call timed out after {self.timeout}s")
            raise Exception(f"API call timed out after {self.timeout}s")
        
        except Exception as e:
            logger.exception("API call failed")
            raise
    
    def generate_sync(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """Synchronous generate for convenience."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        response = loop.run_until_complete(
            self.generate(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        
        return response.content
    
    async def generate_json(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.3,  # Lower temp for structured output
        logprobs: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate JSON response.
        
        Args:
            prompt: User prompt
            system: System message
            temperature: Temperature
            logprobs: Whether to calculate entropy
            
        Returns:
            Parsed JSON dict (includes entropy info if requested)
        """
        # Add JSON instruction
        json_system = (system or "") + "\n\nRespond with valid JSON only. No explanations, just the JSON object."
        json_prompt = prompt
        
        response = await self.generate(
            prompt=json_prompt,
            system=json_system,
            temperature=temperature,
            logprobs=logprobs,
        )
        
        # Parse JSON
        content = response.content.strip()
        logger.debug(f"LLM response content (len={len(content)}): {content[:500]}...")
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Find the closing ```
            if "```" in lines[-1]:
                content = "\n".join(lines[1:-1])
            else:
                # Find the last line with ```
                for i, line in enumerate(lines):
                    if i > 0 and line.strip() == "```":
                        content = "\n".join(lines[1:i])
                        break
                else:
                    content = "\n".join(lines[1:])
        
        content = content.strip()
        
        result = {}
        
        # Try to find JSON in response
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed: {e}")
            
            # Try to extract JSON object
            import re
            
            # Find all potential JSON objects
            json_pattern = r'\{[^{}]*\}'
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        result = parsed
                        break
                except:
                    continue
            
            # Try a more aggressive pattern for nested JSON
            # Find content between first { and last }
            first_brace = content.find('{')
            last_brace = content.rfind('}')
            
            if first_brace >= 0 and last_brace > first_brace:
                json_str = content[first_brace:last_brace + 1]
                
                # Try to fix truncated JSON
                # Count open braces
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                
                if open_braces > close_braces:
                    # Add missing closing braces
                    json_str += '}' * (open_braces - close_braces)
                
                try:
                    result = json.loads(json_str)
                except:
                    pass
            
            if not result:
                logger.warning(f"Failed to parse JSON (len={len(content)}): {content[:300]}...")
                result = {"error": "Failed to parse JSON", "raw": content[:1000]}
        
        # Add entropy info if requested
        if logprobs:
            result["_entropy"] = response.entropy
            result["_confidence"] = response.confidence
            result["_decision"] = response.entropy_decision
        
        return result
    
    async def generate_with_entropy(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
    ) -> LLMResponse:
        """
        Generate response with entropy analysis.
        
        Use this for branch decision: low entropy = 1 branch, high entropy = 2 branches.
        
        Args:
            prompt: User prompt
            system: System message
            temperature: Override temperature
            
        Returns:
            LLMResponse with entropy info populated
        """
        return await self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature or self.temperature,
            logprobs=True,
            top_logprobs=5,
        )


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing.
    
    Returns predefined responses.
    """
    
    def __init__(self, responses: List[str] = None):
        super().__init__(api_key="mock")
        self.responses = responses or ["Mock response"]
        self.call_count = 0
    
    async def generate(self, prompt: str, system: str = None, **kwargs) -> LLMResponse:
        """Return mock response."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        
        return LLMResponse(
            content=response,
            model="mock",
            usage={"total_tokens": len(response)},
            latency=0.001,
        )