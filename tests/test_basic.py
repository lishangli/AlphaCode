"""
Tests for MCTS-Agent.
"""

import pytest
import tempfile
import os

from mcts_agent.config import MCTSConfig, LLMConfig
from mcts_agent.core.node import MCTSNode, NodeStatus, Action, EvaluationResult, FeatureCoords
from mcts_agent.core.tree import SearchTree, FeatureGrid, Island
from mcts_agent.tools.executor import ToolExecutor
from mcts_agent.state.git_manager import GitStateManager


class TestConfig:
    """Test configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MCTSConfig()
        
        assert config.max_iterations == 100
        assert config.exploration_weight == 1.41
        assert config.num_islands == 4
    
    def test_llm_config(self):
        """Test LLM configuration."""
        config = LLMConfig(model="gpt-4o")
        
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = MCTSConfig()
        data = config.to_dict()
        
        assert "max_iterations" in data
        assert "llm" in data


class TestNode:
    """Test MCTS node."""
    
    def test_node_creation(self):
        """Test node creation."""
        node = MCTSNode(
            id="test-123",
            commit_hash="abc123",
            session_id="session-1",
        )
        
        assert node.id == "test-123"
        assert node.commit_hash == "abc123"
        assert node.status == NodeStatus.PENDING
    
    def test_node_update_stats(self):
        """Test node statistics update."""
        node = MCTSNode()
        
        node.update_stats(0.8)
        assert node.visits == 1
        assert node.value_avg == 0.8
        
        node.update_stats(0.6)
        assert node.visits == 2
        assert node.value_avg == 0.7
    
    def test_node_serialization(self):
        """Test node serialization."""
        node = MCTSNode(
            id="test-123",
            commit_hash="abc123",
            visits=5,
            value_avg=0.75,
        )
        
        data = node.to_dict()
        restored = MCTSNode.from_dict(data)
        
        assert restored.id == node.id
        assert restored.visits == node.visits
        assert restored.value_avg == node.value_avg


class TestFeatureCoords:
    """Test feature coordinates."""
    
    def test_coords_creation(self):
        """Test coordinate creation."""
        coords = FeatureCoords(complexity=3, approach=2, quality_tier=7)
        
        assert coords.complexity == 3
        assert coords.to_key() == "3-2-7"
    
    def test_coords_from_key(self):
        """Test coordinate parsing from key."""
        coords = FeatureCoords.from_key("5-3-8")
        
        assert coords.complexity == 5
        assert coords.approach == 3
        assert coords.quality_tier == 8


class TestFeatureGrid:
    """Test feature grid."""
    
    def test_grid_add(self):
        """Test adding to grid."""
        grid = FeatureGrid()
        node = MCTSNode(
            id="node-1",
            feature_coords=FeatureCoords(3, 2, 5),
            value_avg=0.7,
        )
        
        result = grid.add(node)
        
        assert result is True
        assert "3-2-5" in grid.grid
    
    def test_grid_coverage(self):
        """Test grid coverage calculation."""
        grid = FeatureGrid()
        
        assert grid.coverage() == 0.0
        
        grid.grid["0-0-0"] = "node-1"
        grid.grid["1-1-1"] = "node-2"
        
        coverage = grid.coverage()
        assert coverage > 0


class TestSearchTree:
    """Test search tree."""
    
    def test_tree_creation(self):
        """Test tree creation."""
        tree = SearchTree(
            session_id="test-session",
            goal="Test goal",
        )
        
        assert tree.session_id == "test-session"
        assert tree.goal == "Test goal"
    
    def test_tree_add_node(self):
        """Test adding nodes to tree."""
        tree = SearchTree(session_id="test")
        node = MCTSNode(
            id="root",
            commit_hash="abc",
            value_avg=0.5,
        )
        
        tree.add_node(node)
        
        assert "root" in tree.nodes
        assert tree.best_node_id == "root"


class TestToolExecutor:
    """Test tool executor."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = ToolExecutor(root_path=self.temp_dir)
    
    def test_write_and_read(self):
        """Test write and read tools."""
        # Write
        result = self.executor.execute({
            "tool": "write",
            "args": {
                "path": "test.txt",
                "content": "Hello, World!"
            }
        })
        
        assert result.success
        
        # Read
        result = self.executor.execute({
            "tool": "read",
            "args": {"path": "test.txt"}
        })
        
        assert result.success
        assert "Hello, World!" in result.output
    
    def test_edit(self):
        """Test edit tool."""
        # Create file
        self.executor.execute({
            "tool": "write",
            "args": {
                "path": "edit_test.txt",
                "content": "Hello, World!"
            }
        })
        
        # Edit
        result = self.executor.execute({
            "tool": "edit",
            "args": {
                "path": "edit_test.txt",
                "old": "World",
                "new": "MCTS"
            }
        })
        
        assert result.success
        
        # Verify
        result = self.executor.execute({
            "tool": "read",
            "args": {"path": "edit_test.txt"}
        })
        
        assert "MCTS" in result.output


class TestAction:
    """Test action."""
    
    def test_action_creation(self):
        """Test action creation."""
        action = Action(
            description="Add docstring",
            tool_calls=[
                {"tool": "edit", "args": {"path": "test.py", "old": "def ", "new": '"""Doc"""\ndef '}}
            ]
        )
        
        assert action.description == "Add docstring"
        assert len(action.tool_calls) == 1
    
    def test_action_serialization(self):
        """Test action serialization."""
        action = Action(
            id="act-1",
            description="Test action",
            success=True,
        )
        
        data = action.to_dict()
        restored = Action.from_dict(data)
        
        assert restored.id == action.id
        assert restored.description == action.description


class TestEvaluationResult:
    """Test evaluation result."""
    
    def test_result_creation(self):
        """Test result creation."""
        result = EvaluationResult(
            score=0.85,
            metrics={"tests": 0.9, "quality": 0.8},
            level=2,
        )
        
        assert result.score == 0.85
        assert result.is_valid()
    
    def test_result_serialization(self):
        """Test result serialization."""
        result = EvaluationResult(
            score=0.75,
            metrics={"syntax": 1.0, "tests": 0.8},
            artifacts={"errors": []},
        )
        
        data = result.to_dict()
        restored = EvaluationResult.from_dict(data)
        
        assert restored.score == result.score
        assert restored.metrics == result.metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])