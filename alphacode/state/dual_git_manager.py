"""
Dual Git Manager for ALPHACODE.

Separates conversation history Git from code exploration Git.
- Conversation Git: tracks session messages and metadata
- Code Git: tracks code exploration via MCTS

Design principles:
- Minimal, clean separation
- Reuse existing GitStateManager
- Configurable paths
"""

import logging
import os
from pathlib import Path

from alphacode.state.git_manager import GitStateManager

logger = logging.getLogger(__name__)


class DualGitManager:
    """
    Dual Git manager for ALPHACODE.
    
    Manages two separate Git repositories:
    1. Conversation Git: .alphacode/conversation-git/ - tracks session history
    2. Code Git: .alphacode/code-git/ or working directory - tracks code exploration
    
    This separation ensures:
    - Code exploration doesn't pollute user's project Git
    - Session history is preserved independently
    - User's original Git remains untouched during exploration
    """
    
    def __init__(
        self,
        root_path: str = None,
        conversation_subdir: str = ".alphacode/conversation-git",
        code_subdir: str = ".alphacode/code-git",
        use_separate_code_git: bool = True,
    ):
        """
        Initialize dual Git manager.
        
        Args:
            root_path: Project root path
            conversation_subdir: Subdirectory for conversation Git
            code_subdir: Subdirectory for code Git (if use_separate_code_git=True)
            use_separate_code_git: If False, use user's project Git for code
        """
        self.root_path = Path(root_path or os.getcwd()).resolve()
        self.use_separate_code_git = use_separate_code_git
        
        # Conversation Git path
        self.conversation_path = self.root_path / conversation_subdir
        self.conversation_path.mkdir(parents=True, exist_ok=True)
        
        # Code Git path
        if use_separate_code_git:
            self.code_path = self.root_path / code_subdir
            self.code_path.mkdir(parents=True, exist_ok=True)
        else:
            self.code_path = self.root_path
        
        # Initialize Git managers
        self.conversation_git = GitStateManager(
            root_path=str(self.conversation_path),
            branch_prefix="session",
            auto_init=True,
        )
        
        self.code_git = GitStateManager(
            root_path=str(self.code_path),
            branch_prefix="mcts",
            auto_init=True,
        )
        
        logger.debug(f"DualGitManager initialized:")
        logger.debug(f"  Conversation: {self.conversation_path}")
        logger.debug(f"  Code: {self.code_path}")
    
    @property
    def conversation(self) -> GitStateManager:
        """Get conversation Git manager."""
        return self.conversation_git
    
    @property
    def code(self) -> GitStateManager:
        """Get code Git manager."""
        return self.code_git
    
    def get_conversation_path(self) -> Path:
        """Get conversation Git directory path."""
        return self.conversation_path
    
    def get_code_path(self) -> Path:
        """Get code Git directory path."""
        return self.code_path
    
    def cleanup_old_sessions(self, keep_last: int = 10):
        """
        Clean up old session branches, keeping only the most recent ones.
        
        Args:
            keep_last: Number of recent sessions to keep
        """
        try:
            # Get all session branches sorted by commit date
            import subprocess
            result = subprocess.run(
                ["git", "branch", "--list", "session/*", "--sort=-committerdate"],
                capture_output=True,
                text=True,
                cwd=str(self.conversation_path),
            )
            
            branches = [b.strip().lstrip("* ") for b in result.stdout.strip().split("\n") if b.strip()]
            
            # Delete old branches
            for branch in branches[keep_last:]:
                subprocess.run(
                    ["git", "branch", "-D", branch],
                    capture_output=True,
                    cwd=str(self.conversation_path),
                )
                logger.debug(f"Cleaned up old session branch: {branch}")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup old sessions: {e}")
    
    def get_session_count(self) -> int:
        """Get number of session branches."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "branch", "--list", "session/*"],
                capture_output=True,
                text=True,
                cwd=str(self.conversation_path),
            )
            return len([b for b in result.stdout.strip().split("\n") if b.strip()])
        except Exception:
            return 0
    
    def get_exploration_count(self) -> int:
        """Get number of MCTS exploration branches."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "branch", "--list", "mcts/*"],
                capture_output=True,
                text=True,
                cwd=str(self.code_path),
            )
            return len([b for b in result.stdout.strip().split("\n") if b.strip()])
        except Exception:
            return 0
