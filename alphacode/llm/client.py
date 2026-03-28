"""
LLM client for ALPHACODE.

Uses OpenAI SDK for better reliability:
- Built-in retry with exponential backoff
- Connection pooling
- Streaming support
- Proper error handling
"""

import os
import json
import logging
import asyncio
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

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
    entropy_decision: str = "multi"
    # Cache info
    from_cache: bool = False


class LLMClient:
    """
    LLM client using OpenAI SDK.
    
    Features:
    - Built-in retry with exponential backoff
    - Connection pooling
    - Request caching (optional)
    - Environment variable API key
    """
    
    # Shared client instance (connection pool)
    _client: Optional[AsyncOpenAI] = None
    _cache: Optional[Any] = None
    
    def __init__(
        self,
        api_key: str = None,
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_retries: int = 3,
        enable_cache: bool = True,
        cache_ttl: int = 3600,  # 1 hour
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: API key (defaults to env vars: NVIDIA_API_KEY, OPENAI_API_KEY)
            api_base: API base URL
            model: Model name
            temperature: Generation temperature
            max_tokens: Maximum tokens
            timeout: Request timeout in seconds
            max_retries: Max retry attempts
            enable_cache: Enable request caching
            cache_ttl: Cache TTL in seconds
        """
        # API key priority: param > NVIDIA_API_KEY > OPENAI_API_KEY
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        
        if not self.api_key:
            logger.warning("No API key. Set NVIDIA_API_KEY or OPENAI_API_KEY env var.")
        
        # Initialize cache
        if enable_cache and LLMClient._cache is None:
            try:
                from diskcache import Cache
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "alphacode")
                os.makedirs(cache_dir, exist_ok=True)
                LLMClient._cache = Cache(cache_dir)
                logger.info(f"LLM cache enabled: {cache_dir}")
            except ImportError:
                logger.warning("diskcache not installed, caching disabled")
                self.enable_cache = False
    
    def _get_client(self) -> AsyncOpenAI:
        """Get or create shared OpenAI client."""
        if LLMClient._client is None:
            LLMClient._client = AsyncOpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return LLMClient._client
    
    def _cache_key(self, prompt: str, system: str, temperature: float, logprobs: bool) -> str:
        """Generate cache key."""
        key_data = f"{prompt}|{system}|{temperature}|{logprobs}|{self.model}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[LLMResponse]:
        """Get response from cache."""
        if not self.enable_cache or LLMClient._cache is None:
            return None
        try:
            cached = LLMClient._cache.get(key)
            if cached:
                cached.from_cache = True
                logger.debug(f"Cache hit: {key[:8]}")
            return cached
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
            return None
    
    def _save_to_cache(self, key: str, response: LLMResponse):
        """Save response to cache."""
        if not self.enable_cache or LLMClient._cache is None:
            return
        try:
            LLMClient._cache.set(key, response, expire=self.cache_ttl)
            logger.debug(f"Cache saved: {key[:8]}")
        except Exception as e:
            logger.debug(f"Cache write error: {e}")
    
    async def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> LLMResponse:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            system: System message
            temperature: Override temperature
            max_tokens: Override max tokens
            logprobs: Whether to return logprobs
            top_logprobs: Number of top logprobs
            
        Returns:
            LLMResponse
        """
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        # Check cache
        cache_key = self._cache_key(prompt, system or "", temperature, logprobs)
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        start_time = time.time()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            client = self._get_client()
            
            # Build request kwargs
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            if logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = top_logprobs
            
            response = await client.chat.completions.create(**kwargs)
            
            choice = response.choices[0]
            content = choice.message.content
            
            # Calculate entropy if logprobs available
            entropy = 0.5
            confidence = 0.5
            token_logprobs = None
            entropy_decision = "multi"
            
            if logprobs and choice.logprobs and choice.logprobs.content:
                from alphacode.llm.entropy import EntropyAnalyzer
                
                logprobs_content = choice.logprobs.content
                token_logprobs = [t.logprob for t in logprobs_content]
                
                analyzer = EntropyAnalyzer()
                entropy_result = analyzer.analyze(logprobs_content=logprobs_content)
                
                entropy = entropy_result.entropy
                confidence = entropy_result.confidence
                entropy_decision = entropy_result.decision
            
            result = LLMResponse(
                content=content,
                model=response.model or self.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                latency=time.time() - start_time,
                entropy=entropy,
                confidence=confidence,
                logprobs=token_logprobs,
                entropy_decision=entropy_decision,
            )
            
            # Save to cache
            self._save_to_cache(cache_key, result)
            
            return result
            
        except RateLimitError as e:
            logger.error(f"Rate limit hit: {e}")
            raise Exception(f"Rate limit exceeded, please wait: {e}")
        except APITimeoutError as e:
            logger.error(f"API timeout after {self.timeout}s")
            raise Exception(f"Request timed out after {self.timeout}s")
        except APIError as e:
            logger.error(f"API error: {e}")
            raise Exception(f"API error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        """
        Generate streaming response.
        
        Yields:
            str: Text chunks
        """
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        client = self._get_client()
        
        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
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
        temperature: float = 0.3,
        logprobs: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate JSON response.
        """
        json_system = (system or "") + "\n\nRespond with valid JSON only. No explanations."
        
        response = await self.generate(
            prompt=prompt,
            system=json_system,
            temperature=temperature,
            logprobs=logprobs,
        )
        
        content = response.content.strip()
        
        # Remove markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            if "```" in lines[-1]:
                content = "\n".join(lines[1:-1])
            else:
                for i, line in enumerate(lines):
                    if i > 0 and line.strip() == "```":
                        content = "\n".join(lines[1:i])
                        break
                else:
                    content = "\n".join(lines[1:])
        
        content = content.strip()
        
        # Parse JSON
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON
            import re
            first_brace = content.find('{')
            last_brace = content.rfind('}')
            
            if first_brace >= 0 and last_brace > first_brace:
                json_str = content[first_brace:last_brace + 1]
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                if open_braces > close_braces:
                    json_str += '}' * (open_braces - close_braces)
                try:
                    result = json.loads(json_str)
                except:
                    result = {"error": "Parse failed", "raw": content[:500]}
            else:
                result = {"error": "No JSON found", "raw": content[:500]}
        
        # Add entropy info
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
        """Generate response with entropy analysis."""
        return await self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature or self.temperature,
            logprobs=True,
            top_logprobs=5,
        )
    
    @classmethod
    def clear_cache(cls):
        """Clear the LLM response cache."""
        if cls._cache is not None:
            cls._cache.clear()
            logger.info("LLM cache cleared")
    
    @classmethod
    def close(cls):
        """Close the shared client."""
        if cls._client is not None:
            # AsyncOpenAI doesn't have explicit close, but we can reset
            cls._client = None
            logger.debug("LLM client reset")


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""
    
    def __init__(self, responses: List[str] = None):
        super().__init__(api_key="mock", enable_cache=False)
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