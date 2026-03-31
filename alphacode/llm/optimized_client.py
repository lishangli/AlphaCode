"""
Performance optimizations for LLM client.

Key principle: All optimizations are technical (connection pooling, 
parallel processing), NOT bypassing LLM decisions.
"""

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Callable

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class ConnectionWarmup:
    """
    Connection warmup utility.
    
    Pre-establishes HTTP connections to reduce first-request latency.
    This is pure technical optimization, doesn't affect LLM decisions.
    """
    
    def __init__(self, client: AsyncOpenAI, warmup_prompt: str = "hi"):
        """
        Initialize warmup utility.
        
        Args:
            client: OpenAI client
            warmup_prompt: Lightweight prompt for warmup
        """
        self.client = client
        self.warmup_prompt = warmup_prompt
        self._warmed_up = False
        self._warmup_time = 0.0
    
    async def warmup(self) -> float:
        """
        Warm up connection with a lightweight request.
        
        Returns:
            Warmup time in seconds
        """
        if self._warmed_up:
            return self._warmup_time
        
        start = time.time()
        try:
            # Send minimal request to establish connection
            await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use faster model for warmup
                messages=[{"role": "user", "content": self.warmup_prompt}],
                max_tokens=5,
            )
            self._warmup_time = time.time() - start
            self._warmed_up = True
            logger.info(f"Connection warmed up in {self._warmup_time:.2f}s")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
            self._warmup_time = time.time() - start
        
        return self._warmup_time
    
    def is_warmed_up(self) -> bool:
        """Check if connection is warmed up."""
        return self._warmed_up


class ParallelProcessor:
    """
    Parallel processing utility for independent operations.
    
    Used for parallel MCTS node evaluation or batch processing.
    All decisions still come from LLM - this just speeds up execution.
    """
    
    def __init__(self, max_concurrent: int = 5):
        """
        Initialize parallel processor.
        
        Args:
            max_concurrent: Maximum concurrent operations
        """
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_parallel(
        self,
        items: list[Any],
        process_fn: Callable[[Any], Any],
        on_result: Callable[[Any, Any], None] = None,
    ) -> list[Any]:
        """
        Process items in parallel with semaphore control.
        
        Args:
            items: Items to process
            process_fn: Function to process each item
            on_result: Callback for each result (item, result)
            
        Returns:
            List of results
        """
        async def process_with_semaphore(item):
            async with self._semaphore:
                result = await process_fn(item)
                if on_result:
                    on_result(item, result)
                return result
        
        tasks = [process_with_semaphore(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        return [
            r if not isinstance(r, Exception) else None
            for r in results
        ]
    
    async def process_with_progress(
        self,
        items: list[Any],
        process_fn: Callable[[Any], Any],
        progress_callback: Callable[[int, int], None] = None,
    ) -> list[Any]:
        """
        Process items in parallel with progress tracking.
        
        Args:
            items: Items to process
            process_fn: Function to process each item
            progress_callback: Callback for progress (completed, total)
            
        Returns:
            List of results
        """
        total = len(items)
        completed = [0]  # Use list to allow mutation in nested function
        results = [None] * total
        
        async def process_one(index, item):
            async with self._semaphore:
                result = await process_fn(item)
                results[index] = result
                completed[0] += 1
                if progress_callback:
                    progress_callback(completed[0], total)
                return result
        
        tasks = [process_one(i, item) for i, item in enumerate(items)]
        await asyncio.gather(*tasks)
        
        return results


class BatchGenerator:
    """
    Batch generation utility.
    
    Generates multiple responses in parallel.
    Each response is independent LLM decision.
    """
    
    def __init__(self, llm_client: Any):
        """
        Initialize batch generator.
        
        Args:
            llm_client: LLM client instance
        """
        self.llm_client = llm_client
        self.parallel_processor = ParallelProcessor(max_concurrent=3)
    
    async def generate_batch(
        self,
        prompts: list[str],
        system: str = None,
        temperature: float = 0.7,
    ) -> list[str]:
        """
        Generate responses for multiple prompts in parallel.
        
        Args:
            prompts: List of prompts
            system: System prompt
            temperature: Generation temperature
            
        Returns:
            List of generated responses
        """
        async def generate_one(prompt):
            response = await self.llm_client.generate(
                prompt=prompt,
                system=system,
                temperature=temperature,
            )
            return response.content
        
        return await self.parallel_processor.process_parallel(
            prompts,
            generate_one,
        )
    
    async def generate_variations(
        self,
        base_prompt: str,
        variations: list[str],
        system: str = None,
    ) -> list[str]:
        """
        Generate variations of a base prompt.
        
        Useful for MCTS: generate multiple candidate solutions.
        
        Args:
            base_prompt: Base prompt
            variations: List of variation hints
            system: System prompt
            
        Returns:
            List of generated variations
        """
        prompts = [
            f"{base_prompt}\n\nVariation hint: {v}"
            for v in variations
        ]
        
        return await self.generate_batch(prompts, system)


class StreamBuffer:
    """
    Buffer for streaming output with pause/resume capability.
    
    Allows user to pause streaming output.
    """
    
    def __init__(self):
        """Initialize stream buffer."""
        self._buffer: list[str] = []
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Initially not paused
    
    def pause(self):
        """Pause streaming."""
        self._paused = True
        self._pause_event.clear()
        logger.debug("Stream paused")
    
    def resume(self):
        """Resume streaming."""
        self._paused = False
        self._pause_event.set()
        logger.debug("Stream resumed")
    
    async def add(self, token: str):
        """
        Add token to buffer, waiting if paused.
        
        Args:
            token: Token to add
        """
        await self._pause_event.wait()
        self._buffer.append(token)
    
    def get_buffered(self) -> str:
        """Get all buffered content."""
        return "".join(self._buffer)
    
    def clear_buffer(self):
        """Clear buffer."""
        self._buffer.clear()
    
    def is_paused(self) -> bool:
        """Check if paused."""
        return self._paused


class PerformanceMonitor:
    """
    Monitor for tracking performance metrics.
    
    Collects latency, throughput, and other metrics.
    """
    
    def __init__(self):
        """Initialize monitor."""
        self._requests: list[dict] = []
        self._total_tokens = 0
        self._total_time = 0.0
    
    def record_request(
        self,
        latency: float,
        tokens: int,
        cached: bool = False,
        tool_calls: int = 0,
    ):
        """
        Record a request.
        
        Args:
            latency: Request latency
            tokens: Token count
            cached: Whether response was cached
            tool_calls: Number of tool calls
        """
        self._requests.append({
            "latency": latency,
            "tokens": tokens,
            "cached": cached,
            "tool_calls": tool_calls,
            "timestamp": time.time(),
        })
        self._total_tokens += tokens
        self._total_time += latency
    
    def get_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            Statistics dict
        """
        if not self._requests:
            return {
                "total_requests": 0,
                "avg_latency": 0,
                "avg_tokens": 0,
                "total_tokens": 0,
                "throughput": 0,
                "cache_rate": 0,
            }
        
        cached_count = sum(1 for r in self._requests if r["cached"])
        avg_latency = self._total_time / len(self._requests)
        avg_tokens = self._total_tokens / len(self._requests)
        
        return {
            "total_requests": len(self._requests),
            "avg_latency": avg_latency,
            "avg_tokens": avg_tokens,
            "total_tokens": self._total_tokens,
            "throughput": self._total_tokens / self._total_time if self._total_time > 0 else 0,
            "cache_rate": cached_count / len(self._requests),
        }
    
    def get_recent_requests(self, n: int = 10) -> list[dict]:
        """
        Get recent requests.
        
        Args:
            n: Number of recent requests
            
        Returns:
            List of recent requests
        """
        return self._requests[-n:]


# Integration helper
def create_optimized_client(
    base_url: str,
    api_key: str,
    max_connections: int = 100,
    max_keepalive: int = 20,
) -> AsyncOpenAI:
    """
    Create OpenAI client with connection pooling.
    
    Args:
        base_url: API base URL
        api_key: API key
        max_connections: Maximum connections
        max_keepalive: Maximum keepalive connections
        
    Returns:
        Optimized AsyncOpenAI client
    """
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
        ),
        timeout=httpx.Timeout(120.0, connect=30.0),
    )
    
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=http_client,
    )