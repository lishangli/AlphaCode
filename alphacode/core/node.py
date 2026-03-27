"""
MCTS Node data structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time
import uuid


class NodeStatus(Enum):
    """Node status in the search tree."""
    PENDING = "pending"
    EXPLORING = "exploring"
    EVALUATED = "evaluated"
    PRUNED = "pruned"
    ADOPTED = "adopted"


@dataclass
class FeatureCoords:
    """
    MAP-Elites feature coordinates.
    
    Represents a position in the feature space.
    """
    complexity: int = 0        # Complexity bin (0-9)
    approach: int = 0          # Approach type bin (0-9)
    quality_tier: int = 0      # Quality tier bin (0-9)
    
    def to_tuple(self) -> tuple:
        """Convert to tuple for use as dict key."""
        return (self.complexity, self.approach, self.quality_tier)
    
    def to_key(self) -> str:
        """Convert to string key."""
        return f"{self.complexity}-{self.approach}-{self.quality_tier}"
    
    @classmethod
    def from_key(cls, key: str) -> "FeatureCoords":
        """Create from string key."""
        parts = key.split("-")
        return cls(
            complexity=int(parts[0]),
            approach=int(parts[1]),
            quality_tier=int(parts[2]),
        )


@dataclass
class Action:
    """
    Composite action: a group of related tool calls.
    
    Design principle:
    - One action = one meaningful improvement attempt
    - Not a single tool call, but a coherent set of operations
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    
    # Execution results
    success: bool = False
    error: Optional[str] = None
    
    # Confidence score (0.0-1.0) for smart execution ordering
    confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "tool_calls": self.tool_calls,
            "reasoning": self.reasoning,
            "success": self.success,
            "error": self.error,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            description=data.get("description", ""),
            tool_calls=data.get("tool_calls", []),
            reasoning=data.get("reasoning", ""),
            success=data.get("success", False),
            error=data.get("error"),
            confidence=data.get("confidence", 0.5),
        )


@dataclass
class EvaluationResult:
    """
    Evaluation result for a node.
    
    Contains:
    - Main score
    - Individual metrics
    - Artifacts (error messages, outputs, etc.)
    """
    # Main score
    score: float = 0.0
    
    # Individual metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Artifacts
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation level (cascade)
    level: int = 0  # 0=syntax, 1=tests, 2=quality, 3=integration
    
    def is_valid(self) -> bool:
        """Check if result is valid."""
        return self.score > 0 or len(self.metrics) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "level": self.level,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary."""
        return cls(
            score=data.get("score", 0.0),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", {}),
            level=data.get("level", 0),
        )


@dataclass
class MCTSNode:
    """
    MCTS search tree node.
    
    Core design:
    - State represented by Git commit hash
    - Contains MCTS statistics
    - Supports MAP-Elites feature grid
    """
    # ========== Identity ==========
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    commit_hash: str = ""
    session_id: str = ""
    
    # ========== Tree Structure ==========
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    
    # ========== Action ==========
    action: Optional[Action] = None
    
    # ========== MCTS Statistics ==========
    visits: int = 0
    value_sum: float = 0.0
    value_avg: float = 0.0
    
    # UCB cache
    ucb_score: float = 0.0
    
    # ========== Evaluation ==========
    evaluation: Optional[EvaluationResult] = None
    status: NodeStatus = NodeStatus.PENDING
    
    # ========== MAP-Elites ==========
    feature_coords: Optional[FeatureCoords] = None
    
    # ========== Island ==========
    island_id: int = 0
    
    # ========== Metadata ==========
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # ========== Code Cache ==========
    _code_cache: Optional[str] = field(default=None, repr=False)
    
    # ========== Artifacts ==========
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def code(self) -> str:
        """Get node code (requires GitStateManager to populate cache)."""
        return self._code_cache or ""
    
    @code.setter
    def code(self, value: str):
        """Set code cache."""
        self._code_cache = value
    
    def update_stats(self, value: float):
        """Update MCTS statistics."""
        self.visits += 1
        self.value_sum += value
        self.value_avg = self.value_sum / self.visits
        self.updated_at = time.time()
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children)."""
        return len(self.children_ids) == 0
    
    def is_root(self) -> bool:
        """Check if node is root (no parent)."""
        return self.parent_id is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "commit_hash": self.commit_hash,
            "session_id": self.session_id,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "depth": self.depth,
            "action": self.action.to_dict() if self.action else None,
            "visits": self.visits,
            "value_sum": self.value_sum,
            "value_avg": self.value_avg,
            "ucb_score": self.ucb_score,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "status": self.status.value,
            "feature_coords": self.feature_coords.to_key() if self.feature_coords else None,
            "island_id": self.island_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "artifacts": self.artifacts,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCTSNode":
        """Create from dictionary."""
        node = cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            commit_hash=data.get("commit_hash", ""),
            session_id=data.get("session_id", ""),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            depth=data.get("depth", 0),
            visits=data.get("visits", 0),
            value_sum=data.get("value_sum", 0.0),
            value_avg=data.get("value_avg", 0.0),
            ucb_score=data.get("ucb_score", 0.0),
            status=NodeStatus(data.get("status", "pending")),
            island_id=data.get("island_id", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            artifacts=data.get("artifacts", {}),
        )
        
        # Restore action
        if data.get("action"):
            node.action = Action.from_dict(data["action"])
        
        # Restore evaluation
        if data.get("evaluation"):
            node.evaluation = EvaluationResult.from_dict(data["evaluation"])
        
        # Restore feature coords
        if data.get("feature_coords"):
            node.feature_coords = FeatureCoords.from_key(data["feature_coords"])
        
        return node