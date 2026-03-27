"""
Entropy-based confidence estimation for LLM outputs.

Uses model's logprobs to measure uncertainty and decide exploration strategy.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class EntropyResult:
    """
    Result of entropy analysis.
    
    Attributes:
        entropy: Average entropy across tokens (lower = more confident)
        normalized_entropy: Entropy normalized to [0, 1] range
        confidence: Inverse of normalized entropy (higher = more confident)
        token_entropies: Per-token entropy values
        decision: "single" or "multi" based on entropy threshold
        reasoning: Explanation of the decision
    """
    entropy: float
    normalized_entropy: float
    confidence: float
    token_entropies: List[float] = field(default_factory=list)
    decision: str = "multi"
    reasoning: str = ""
    
    def __str__(self) -> str:
        return (
            f"EntropyResult(entropy={self.entropy:.3f}, "
            f"confidence={self.confidence:.3f}, decision={self.decision})"
        )


class EntropyAnalyzer:
    """
    Analyzes LLM output entropy to determine confidence.
    
    Key insight:
    - Low entropy = model is certain = use single branch (save compute)
    - High entropy = model is uncertain = use multi branch (explore more)
    
    Entropy formula:
        H = -Σ p(x) * log(p(x))
    
    For logprobs from API (which are log base e):
        H = -Σ exp(logprob) * logprob
    
    Observed entropy ranges (from testing):
    - Very confident: 0.01 - 0.05
    - Medium confidence: 0.05 - 0.10  
    - Uncertain: > 0.10
    
    Perplexity ranges:
    - Confident: 1.0 - 1.1
    - Medium: 1.1 - 1.2
    - Uncertain: > 1.2
    """
    
    # Thresholds based on observed entropy values
    LOW_ENTROPY_THRESHOLD = 0.05   # Below this: very confident, single branch
    HIGH_ENTROPY_THRESHOLD = 0.10  # Above this: uncertain, multi branch
    
    # Perplexity thresholds (alternative measure)
    LOW_PERPLEXITY_THRESHOLD = 1.1
    HIGH_PERPLEXITY_THRESHOLD = 1.2
    
    def __init__(
        self,
        low_threshold: float = 0.05,
        high_threshold: float = 0.10,
        vocab_size: int = 32000,
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.max_entropy = math.log(vocab_size)
    
    def calculate_token_entropy(self, logprob: float) -> float:
        """
        Calculate entropy for a single token.
        
        For a single token with logprob, we estimate entropy as:
        H ≈ -exp(logprob) * logprob
        
        This is actually the surprisal/energy. For true entropy we'd need
        the full distribution, but top_logprobs gives us enough to estimate.
        
        Args:
            logprob: Natural log of probability
            
        Returns:
            Estimated entropy contribution
        """
        prob = math.exp(logprob)
        if prob <= 0 or prob >= 1:
            return 0.0
        return -prob * logprob
    
    def calculate_entropy_from_logprobs(
        self,
        token_logprobs: List[float],
    ) -> float:
        """
        Calculate actual entropy from token logprobs.
        
        Uses the true entropy formula: H = -Σ p(x) * log(p(x))
        
        Args:
            token_logprobs: List of log probabilities for each token
            
        Returns:
            Average entropy per token
        """
        if not token_logprobs:
            return 0.05  # Default to medium
        
        entropies = []
        for logprob in token_logprobs:
            prob = math.exp(logprob)
            if prob > 0.001:  # Avoid log(0)
                # H = -p * log(p)
                entropy_contrib = -prob * logprob
                entropies.append(entropy_contrib)
            else:
                entropies.append(0.0)
        
        return sum(entropies) / len(entropies)
    
    def calculate_perplexity(self, token_logprobs: List[float]) -> float:
        """
        Calculate perplexity from logprobs.
        
        Perplexity = exp(-avg_logprob)
        
        Lower perplexity = more confident.
        Higher perplexity = more uncertain.
        
        Args:
            token_logprobs: List of log probabilities
            
        Returns:
            Perplexity value
        """
        if not token_logprobs:
            return 1.5  # Default
        
        avg_logprob = sum(token_logprobs) / len(token_logprobs)
        return math.exp(-avg_logprob)
    
    def analyze(
        self,
        logprobs_content: Optional[List[Any]] = None,
        token_logprobs: Optional[List[float]] = None,
    ) -> EntropyResult:
        """
        Analyze entropy and determine exploration strategy.
        
        Uses both entropy and perplexity for robust decision making.
        
        Args:
            logprobs_content: OpenAI-style logprobs content list
            token_logprobs: Alternative: list of logprob floats
            
        Returns:
            EntropyResult with confidence and decision
        """
        # Extract logprobs if given OpenAI format
        if logprobs_content and not token_logprobs:
            token_logprobs = []
            for token_info in logprobs_content:
                if hasattr(token_info, 'logprob'):
                    token_logprobs.append(token_info.logprob)
                elif isinstance(token_info, dict):
                    token_logprobs.append(token_info.get('logprob', 0))
        
        if not token_logprobs:
            # No data, default to multi-branch
            return EntropyResult(
                entropy=0.05,
                normalized_entropy=0.5,
                confidence=0.5,
                decision="multi",
                reasoning="No logprobs available, defaulting to multi-branch",
            )
        
        # Calculate actual entropy
        entropy = self.calculate_entropy_from_logprobs(token_logprobs)
        
        # Calculate perplexity as alternative measure
        perplexity = self.calculate_perplexity(token_logprobs)
        
        # Normalize entropy to [0, 1] based on observed range
        # Observed range: 0.01 - 0.15
        # Normalize: entropy / 0.15 (cap at 1.0)
        normalized_entropy = min(entropy / 0.15, 1.0)
        confidence = 1.0 - normalized_entropy
        
        # Make decision using BOTH entropy and perplexity
        # Decision is "single" only if both metrics agree
        low_entropy = entropy < self.low_threshold
        low_perplexity = perplexity < self.LOW_PERPLEXITY_THRESHOLD
        
        high_entropy = entropy > self.high_threshold
        high_perplexity = perplexity > self.HIGH_PERPLEXITY_THRESHOLD
        
        if low_entropy and low_perplexity:
            # Very confident - single branch
            decision = "single"
            reasoning = (
                f"High confidence: entropy={entropy:.4f} (< {self.low_threshold}), "
                f"perplexity={perplexity:.3f} (< {self.LOW_PERPLEXITY_THRESHOLD}). "
                f"Using single branch."
            )
        elif high_entropy or high_perplexity:
            # Uncertain - multi branch
            decision = "multi"
            reasoning = (
                f"Uncertainty detected: entropy={entropy:.4f}, "
                f"perplexity={perplexity:.3f}. "
                f"Using multi-branch exploration."
            )
        else:
            # Medium confidence - default to multi for safety
            decision = "multi"
            reasoning = (
                f"Medium confidence: entropy={entropy:.4f}, "
                f"perplexity={perplexity:.3f}. "
                f"Using multi-branch as precaution."
            )
        
        return EntropyResult(
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            confidence=confidence,
            token_entropies=token_logprobs,
            decision=decision,
            reasoning=reasoning,
        )
    
    def should_explore_multi(self, entropy_result: EntropyResult) -> bool:
        """
        Decide if multi-branch exploration is needed.
        
        Args:
            entropy_result: Result from analyze()
            
        Returns:
            True if should use multi-branch
        """
        return entropy_result.decision == "multi"
    
    def get_num_branches(
        self,
        entropy_result: EntropyResult,
        max_branches: int = 2,
    ) -> int:
        """
        Determine number of branches based on entropy.
        
        Low entropy: 1 branch (model is confident)
        High entropy: 2+ branches (model is uncertain)
        
        Args:
            entropy_result: Result from analyze()
            max_branches: Maximum branches to generate
            
        Returns:
            Number of branches to generate
        """
        if entropy_result.decision == "single":
            return 1
        else:
            # Scale branches by uncertainty
            # More uncertainty = more branches (up to max)
            uncertainty = entropy_result.normalized_entropy
            # 0.5 entropy -> 2 branches, 0.7 -> 2, 0.9 -> maybe 3
            if uncertainty > 0.8 and max_branches >= 3:
                return 3
            return 2


# Convenience function
def analyze_entropy(logprobs_content: List[Any]) -> EntropyResult:
    """
    Analyze entropy from OpenAI-style logprobs.
    
    Args:
        logprobs_content: List of token info with logprob field
        
    Returns:
        EntropyResult with confidence and decision
    """
    analyzer = EntropyAnalyzer()
    return analyzer.analyze(logprobs_content=logprobs_content)