"""
Smart cache system - LLM-driven similarity judgment.

Key principle: Cache decisions are made by LLM, not hardcoded rules.
This maintains maximum flexibility.
"""

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class CachedEntry:
    """A cached response entry."""
    query: str
    answer: str
    timestamp: float
    metadata: dict = field(default_factory=dict)
    # Similarity judgment context
    context: str = ""
    # Usage count
    use_count: int = 0


class SmartCache:
    """
    Intelligent cache with LLM-driven similarity judgment.

    Design principle:
    - Cache retrieval and similarity judgment are done by LLM
    - LLM decides if cached answer can be reused or needs adaptation
    - No hardcoded rules - maximum flexibility
    """

    SIMILARITY_PROMPT = """判断以下两个问题是否本质相同,可以用相同或相似的回答解决:

问题1: {query1}
问题2: {query2}

考虑因素:
1. 核心意图是否相同
2. 所需的代码/解决方案是否相似
3. 是否可以用相同的方法处理

只回答 YES 或 NO,不要解释。"""

    ADAPT_PROMPT = """用户提出了一个与之前问题相似的新问题。请根据新问题调整之前的回答。

之前的问题: {original_query}
之前的回答: {original_answer}

新问题: {new_query}

请调整回答以更好地匹配新问题。保持回答的核心内容,但根据新问题的具体要求进行修改。"""

    def __init__(
        self,
        llm_client: Any,
        cache_dir: str = None,
        ttl: int = 3600,  # 1 hour
        similarity_threshold: float = 0.85,
        enable_llm_judgment: bool = True,
        min_keyword_sim: float = 0.3,  # Minimum for candidate filtering
    ):
        """
        Initialize smart cache.

        Args:
            llm_client: LLM client for similarity judgment
            cache_dir: Directory for persistent cache
            ttl: Cache entry TTL in seconds
            similarity_threshold: Minimum similarity score to reuse
            enable_llm_judgment: Use LLM for similarity judgment
            min_keyword_sim: Minimum keyword similarity for candidate filtering
        """
        self.llm_client = llm_client
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold
        self.enable_llm_judgment = enable_llm_judgment
        self.min_keyword_sim = min_keyword_sim

        # Cache storage
        self._cache: dict[str, CachedEntry] = {}
        self._cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "alphacode", "smart_cache"
        )

        # Stats
        self._hits = 0
        self._misses = 0
        self._llm_judgments = 0

        # Ensure cache directory exists
        os.makedirs(self._cache_dir, exist_ok=True)

        logger.info(f"SmartCache initialized: {self._cache_dir}")

    def _hash_query(self, query: str) -> str:
        """Generate hash for query."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _keyword_similarity(self, query1: str, query2: str) -> float:
        """
        Quick keyword-based similarity check (first pass).

        This is a fast heuristic, not the final decision.
        LLM makes the final similarity judgment.
        """
        # Normalize and tokenize (support Chinese)
        words1 = set(query1.lower().replace("", " ").split())
        words2 = set(query2.lower().replace("", " ").split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _find_candidates(self, query: str) -> list[CachedEntry]:
        """
        Find candidate cached entries based on quick keyword similarity.
        
        This is just a first pass filter. LLM makes the final decision.
        
        Args:
            query: New query
            
        Returns:
            List of candidate entries (ordered by similarity)
        """
        candidates = []
        
        for entry in self._cache.values():
            # Check TTL
            if time.time() - entry.timestamp > self.ttl:
                continue
            
            # Quick keyword check (just for filtering, not final decision)
            sim = self._keyword_similarity(query, entry.query)
            if sim >= self.min_keyword_sim:
                candidates.append((sim, entry))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        return [entry for _, entry in candidates[:5]]  # Top 5 candidates

    async def _llm_judge_similarity(self, query1: str, query2: str) -> bool:
        """
        Let LLM judge if two queries are similar.

        This is the key decision - made by LLM, not hardcoded rules.

        Args:
            query1: First query
            query2: Second query

        Returns:
            True if LLM judges them similar
        """
        if not self.enable_llm_judgment:
            # Without LLM, use keyword similarity threshold
            return self._keyword_similarity(query1, query2) >= self.similarity_threshold

        self._llm_judgments += 1

        prompt = self.SIMILARITY_PROMPT.format(
            query1=query1,
            query2=query2
        )

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,  # Low temperature for judgment
                max_tokens=10,
            )

            result = "YES" in response.content.upper()
            logger.debug(f"LLM similarity judgment: {query1[:30]} vs {query2[:30]} -> {result}")
            return result

        except Exception as e:
            logger.warning(f"LLM similarity judgment failed: {e}")
            # Fallback to keyword similarity
            return self._keyword_similarity(query1, query2) >= self.similarity_threshold

    async def _llm_adapt_answer(
        self,
        original_answer: str,
        original_query: str,
        new_query: str,
    ) -> str:
        """
        Let LLM adapt cached answer to new query.

        Args:
            original_answer: Cached answer
            original_query: Original query
            new_query: New query

        Returns:
            Adapted answer
        """
        prompt = self.ADAPT_PROMPT.format(
            original_query=original_query,
            original_answer=original_answer,
            new_query=new_query,
        )

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=2048,
            )
            return response.content
        except Exception as e:
            logger.warning(f"LLM adaptation failed: {e}")
            return original_answer

    async def get_or_compute(
        self,
        query: str,
        compute_fn: Callable,
        context: str = "",
    ) -> tuple[str, bool, dict]:
        """
        Get cached answer or compute new one.

        LLM decides:
        1. Whether cached answer can be reused
        2. Whether cached answer needs adaptation

        Args:
            query: User query
            compute_fn: Function to compute answer if not cached
            context: Additional context

        Returns:
            Tuple of (answer, from_cache, metadata)
        """
        # 1. Exact match check
        query_hash = self._hash_query(query)
        if query_hash in self._cache:
            entry = self._cache[query_hash]
            if time.time() - entry.timestamp <= self.ttl:
                entry.use_count += 1
                self._hits += 1
                logger.info(f"Cache exact hit: {query[:50]}")
                return entry.answer, True, {"type": "exact_match"}

        # 2. Find candidates (quick keyword filter)
        candidates = self._find_candidates(query)

        if candidates:
            # 3. LLM judges similarity for each candidate
            for candidate in candidates:
                is_similar = await self._llm_judge_similarity(query, candidate.query)

                if is_similar:
                    # 4. LLM decides if answer needs adaptation
                    keyword_sim = self._keyword_similarity(query, candidate.query)

                    if keyword_sim >= 0.9:
                        # Very similar - use directly
                        candidate.use_count += 1
                        self._hits += 1
                        logger.info(f"Cache direct reuse: {query[:50]}")
                        return candidate.answer, True, {"type": "direct_reuse"}
                    else:
                        # Needs adaptation - LLM adapts
                        adapted = await self._llm_adapt_answer(
                            candidate.answer,
                            candidate.query,
                            query,
                        )
                        self._hits += 1
                        logger.info(f"Cache adapted reuse: {query[:50]}")
                        return adapted, True, {"type": "adapted", "original_query": candidate.query}

        # 5. No cache hit - compute new answer
        self._misses += 1
        logger.info(f"Cache miss: {query[:50]}")

        # Support both sync and async compute_fn
        import asyncio
        if asyncio.iscoroutinefunction(compute_fn):
            answer = await compute_fn()
        else:
            # Wrap sync function
            answer = compute_fn()
            if asyncio.iscoroutine(answer):
                answer = await answer

        # 6. Store in cache
        self._cache[query_hash] = CachedEntry(
            query=query,
            answer=answer,
            timestamp=time.time(),
            context=context,
        )

        return answer, False, {"type": "computed"}

    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "llm_judgments": self._llm_judgments,
            "entries": len(self._cache),
        }

    def clear(self):
        """Clear cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._llm_judgments = 0
        logger.info("Cache cleared")

    def save(self):
        """Save cache to disk."""
        import json

        cache_file = os.path.join(self._cache_dir, "cache.json")

        try:
            data = {
                k: {
                    "query": v.query,
                    "answer": v.answer,
                    "timestamp": v.timestamp,
                    "context": v.context,
                    "use_count": v.use_count,
                }
                for k, v in self._cache.items()
            }

            with open(cache_file, "w") as f:
                json.dump(data, f)

            logger.debug(f"Cache saved: {len(data)} entries")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def load(self):
        """Load cache from disk."""
        import json

        cache_file = os.path.join(self._cache_dir, "cache.json")

        if not os.path.exists(cache_file):
            return

        try:
            with open(cache_file) as f:
                data = json.load(f)

            for k, v in data.items():
                self._cache[k] = CachedEntry(
                    query=v["query"],
                    answer=v["answer"],
                    timestamp=v["timestamp"],
                    context=v.get("context", ""),
                    use_count=v.get("use_count", 0),
                )

            # Remove expired entries
            expired = [
                k for k, v in self._cache.items()
                if time.time() - v.timestamp > self.ttl
            ]
            for k in expired:
                del self._cache[k]

            logger.info(f"Cache loaded: {len(self._cache)} valid entries")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")