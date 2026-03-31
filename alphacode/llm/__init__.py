"""
LLM module for MCTS-Agent.
"""

from alphacode.llm.client import LLMClient, LLMResponse
from alphacode.llm.smart_cache import SmartCache, CachedEntry
from alphacode.llm.optimized_client import (
    ConnectionWarmup,
    ParallelProcessor,
    BatchGenerator,
    StreamBuffer,
    PerformanceMonitor,
    create_optimized_client,
)

__all__ = [
    "LLMClient",
    "LLMResponse",
    # Smart cache - LLM decides similarity
    "SmartCache",
    "CachedEntry",
    # Performance - pure technical optimization
    "ConnectionWarmup",
    "ParallelProcessor",
    "BatchGenerator",
    "StreamBuffer",
    "PerformanceMonitor",
    "create_optimized_client",
]
