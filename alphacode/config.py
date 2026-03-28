"""
Configuration for MCTS-Agent.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import os


@dataclass
class LLMConfig:
    """LLM configuration"""
    
    model: str = "meta/llama-3.1-8b-instruct"
    api_key: Optional[str] = None
    api_base: str = "https://integrate.api.nvidia.com/v1"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120
    
    # Retry configuration
    max_retries: int = 3
    
    # Cache configuration
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    # Multi-model ensemble
    models: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        # Load API key from environment if not set
        # Priority: NVIDIA_API_KEY > OPENAI_API_KEY
        if self.api_key is None:
            self.api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")


@dataclass
class MCTSConfig:
    """
    MCTS configuration.
    
    Controls all aspects of the search process.
    """
    
    # ========== Search Configuration ==========
    max_iterations: int = 10
    max_depth: int = 10
    max_nodes: int = 100
    
    # ========== UCB Configuration ==========
    exploration_weight: float = 1.41  # sqrt(2)
    exploration_decay: float = 0.995
    
    # ========== Island Configuration ==========
    num_islands: int = 2
    migration_interval: int = 50
    migration_rate: float = 0.1
    
    # ========== Evaluation Configuration ==========
    evaluation_timeout: int = 60
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.6, 0.8])
    
    # Value function weights
    weight_syntax: float = 0.2
    weight_tests: float = 0.3
    weight_quality: float = 0.2
    weight_progress: float = 0.3
    
    # ========== Action Generation ==========
    num_actions_per_expand: int = 2  # 2分支探索
    max_tool_calls_per_action: int = 5
    
    # ========== Inspiration Configuration ==========
    num_inspirations: int = 2
    inspiration_diversity: bool = True
    
    # ========== Pruning Configuration ==========
    prune_threshold: float = 0.1
    prune_after_visits: int = 3
    
    # ========== Termination Conditions ==========
    target_score: float = 0.85
    convergence_window: int = 10
    convergence_threshold: float = 0.01
    max_retries: int = 2  # 最终测试失败后的最大重试次数
    
    # ========== Intent Detection ==========
    intent_check: bool = True  # 是否启用意图判断
    
    # ========== Parallel Configuration ==========
    parallel_workers: int = 1
    
    # ========== Logging Configuration ==========
    log_level: str = "INFO"
    log_interval: int = 2
    save_checkpoints: bool = True
    checkpoint_interval: int = 5
    
    # ========== Git Configuration ==========
    git_branch_prefix: str = "mcts"
    auto_commit: bool = True
    auto_merge_best: bool = True
    
    # ========== LLM Configuration ==========
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # ========== Feature Grid Configuration ==========
    feature_dimensions: List[str] = field(
        default_factory=lambda: ["complexity", "approach", "quality_tier"]
    )
    feature_bins: int = 10
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "MCTSConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCTSConfig":
        """Create configuration from dictionary."""
        # Handle nested LLM config
        if "llm" in data:
            llm_data = data.pop("llm")
            data["llm"] = LLMConfig(**llm_data)
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)