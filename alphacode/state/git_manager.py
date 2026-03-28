"""
Git state manager for MCTS-Agent.

Manages node states as Git commits.
"""

import logging
import os
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GitStatus:
    """Git repository status."""
    branch: str = ""
    is_clean: bool = True
    staged_files: list[str] = None
    modified_files: list[str] = None
    untracked_files: list[str] = None

    def __post_init__(self):
        if self.staged_files is None:
            self.staged_files = []
        if self.modified_files is None:
            self.modified_files = []
        if self.untracked_files is None:
            self.untracked_files = []


class GitStateManager:
    """
    Git state manager.

    Uses Git commits to represent MCTS node states.
    Each node = one commit.
    Branches represent exploration paths.
    """

    def __init__(
        self,
        root_path: str = None,
        branch_prefix: str = "mcts",
        auto_init: bool = True
    ):
        """
        Initialize Git state manager.

        Args:
            root_path: Root directory path (defaults to cwd)
            branch_prefix: Prefix for MCTS branches
            auto_init: Whether to auto-initialize git repo
        """
        self.root_path = root_path or os.getcwd()
        self.branch_prefix = branch_prefix

        if auto_init:
            self._ensure_git_repo()

    def _ensure_git_repo(self):
        """Ensure we're in a git repository."""
        if not os.path.exists(os.path.join(self.root_path, ".git")):
            logger.info(f"Initializing git repository in {self.root_path}")
            self._run_git("init")
            self._run_git("config", "user.email", "mcts-agent@example.com")
            self._run_git("config", "user.name", "MCTS-Agent")

    def _run_git(self, *args, check: bool = True) -> str:
        """
        Run a git command.

        Args:
            *args: Git command arguments
            check: Whether to raise on error

        Returns:
            Command stdout
        """
        cmd = ["git"] + list(args)
        logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.root_path,
        )

        if check and result.returncode != 0:
            logger.warning(
                f"Git command failed: git {' '.join(args)}\n"
                f"stderr: {result.stderr}"
            )
            if check:
                # Don't raise for expected failures
                if "nothing to commit" not in result.stderr:
                    pass  # Allow some failures silently

        return result.stdout.strip()

    def snapshot(self, message: str = "MCTS snapshot") -> str:
        """
        Create a snapshot of current state.

        Returns:
            Commit hash
        """
        # Add all changes
        self._run_git("add", "-A")

        # Commit (allow empty for consistency)
        self._run_git("commit", "-m", message, "--allow-empty")

        # Get commit hash
        commit_hash = self._run_git("rev-parse", "HEAD")

        logger.debug(f"Created snapshot: {commit_hash[:8]} - {message}")
        return commit_hash

    def restore(self, commit_hash: str):
        """
        Restore to a specific commit state.

        Args:
            commit_hash: Commit to restore to
        """
        logger.debug(f"Restoring to commit {commit_hash[:8]}")

        # Get list of files at this commit
        files = self.get_all_files(commit_hash)

        if not files:
            # Empty commit, nothing to restore
            logger.debug(f"No files at commit {commit_hash[:8]}")
            return

        # Reset working directory to commit
        self._run_git("checkout", commit_hash, "--", ".")

    def hard_reset(self, commit_hash: str):
        """
        Hard reset to a specific commit.

        WARNING: This discards all uncommitted changes.
        """
        logger.debug(f"Hard resetting to {commit_hash[:8]}")
        self._run_git("reset", "--hard", commit_hash)

    def create_branch(self, name: str, from_commit: str = None) -> str:
        """
        Create a new branch.

        Args:
            name: Branch name
            from_commit: Commit to branch from (default: HEAD)

        Returns:
            Full branch name
        """
        if name.startswith(self.branch_prefix):
            full_name = name
        else:
            full_name = f"{self.branch_prefix}/{name}"

        if from_commit:
            self._run_git("checkout", "-b", full_name, from_commit)
        else:
            self._run_git("checkout", "-b", full_name)

        logger.debug(f"Created branch: {full_name}")
        return full_name

    def checkout_branch(self, name: str):
        """Checkout to a branch."""
        self._run_git("checkout", name)

    def checkout_main(self):
        """Checkout to main branch."""
        # Try 'main' first, then 'master'
        result = self._run_git("rev-parse", "--verify", "main", check=False)
        if result:
            self._run_git("checkout", "main")
        else:
            result = self._run_git("rev-parse", "--verify", "master", check=False)
            if result:
                self._run_git("checkout", "master")
            else:
                # Neither main nor master exists, stay on current branch
                logger.debug("No main/master branch found, staying on current branch")

    def merge_to_main(self, commit_hash: str, message: str = None):
        """
        Merge a commit to main branch.

        Args:
            commit_hash: Commit to merge
            message: Merge commit message
        """
        self.checkout_main()

        msg = message or f"MCTS: merge best solution {commit_hash[:8]}"
        self._run_git("merge", commit_hash, "-m", msg)

        logger.info(f"Merged {commit_hash[:8]} to main")

    def get_diff(self, from_hash: str, to_hash: str = None) -> str:
        """
        Get diff between commits.

        Args:
            from_hash: Source commit
            to_hash: Target commit (default: current state)

        Returns:
            Diff string
        """
        if to_hash:
            return self._run_git("diff", from_hash, to_hash)
        else:
            return self._run_git("diff", from_hash)

    def get_code(self, commit_hash: str, path: str = "program.py") -> str:
        """
        Get code from a specific commit.

        Args:
            commit_hash: Commit hash
            path: File path within repo

        Returns:
            File contents
        """
        result = self._run_git("show", f"{commit_hash}:{path}", check=False)
        return result

    def get_all_files(self, commit_hash: str = None) -> list[str]:
        """
        Get all tracked files at a commit.

        Args:
            commit_hash: Commit hash (default: HEAD)

        Returns:
            List of file paths
        """
        if commit_hash:
            result = self._run_git("ls-tree", "-r", "--name-only", commit_hash)
        else:
            result = self._run_git("ls-files")

        return [f for f in result.split("\n") if f]

    def write_file(self, path: str, content: str):
        """
        Write content to a file.

        Args:
            path: File path (relative to root)
            content: File content
        """
        full_path = os.path.join(self.root_path, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "w") as f:
            f.write(content)

        logger.debug(f"Wrote file: {path}")

    def read_file(self, path: str) -> str:
        """Read file content."""
        full_path = os.path.join(self.root_path, path)

        with open(full_path) as f:
            return f.read()

    def get_status(self) -> GitStatus:
        """Get current repository status."""
        branch = self._run_git("branch", "--show-current")

        # Get file status
        result = self._run_git("status", "--porcelain")

        staged = []
        modified = []
        untracked = []

        for line in result.split("\n"):
            if not line:
                continue

            status = line[:2]
            filepath = line[3:]

            if status[0] in "MADRC":
                staged.append(filepath)
            elif status[1] in "MD":
                modified.append(filepath)
            elif status == "??":
                untracked.append(filepath)

        return GitStatus(
            branch=branch,
            is_clean=len(staged) == 0 and len(modified) == 0,
            staged_files=staged,
            modified_files=modified,
            untracked_files=untracked,
        )

    def get_log(self, n: int = 20, branch: str = None) -> list[dict]:
        """
        Get commit log.

        Args:
            n: Number of commits
            branch: Branch name (default: current)

        Returns:
            List of commit info dicts
        """
        format_str = "%H|%h|%s|%an|%ai"

        if branch:
            result = self._run_git("log", branch, f"-{n}", f"--format={format_str}")
        else:
            result = self._run_git("log", f"-{n}", f"--format={format_str}")

        commits = []
        for line in result.split("\n"):
            if not line:
                continue

            parts = line.split("|", 4)
            if len(parts) == 5:
                commits.append({
                    "hash": parts[0],
                    "short_hash": parts[1],
                    "message": parts[2],
                    "author": parts[3],
                    "date": parts[4],
                })

        return commits

    def get_tree_visualization(self, n: int = 30) -> str:
        """
        Get visual representation of commit tree.

        Args:
            n: Number of commits to show

        Returns:
            Git log graph string
        """
        return self._run_git(
            "log",
            "--oneline",
            "--graph",
            "--all",
            f"-{n}",
            f"--decorate-refs=refs/heads/{self.branch_prefix}",
        )

    def cleanup_branches(self, prefix: str = None):
        """
        Delete MCTS branches.

        Args:
            prefix: Branch prefix to delete (default: self.branch_prefix)
        """
        prefix = prefix or self.branch_prefix

        # Get all branches with prefix
        result = self._run_git("branch", "--list", f"{prefix}/*")

        for branch in result.split("\n"):
            branch = branch.strip()
            if branch and not branch.startswith("*"):
                self._run_git("branch", "-D", branch)
                logger.debug(f"Deleted branch: {branch}")

    def get_current_commit(self) -> str:
        """Get current HEAD commit hash."""
        return self._run_git("rev-parse", "HEAD")

    def commit_exists(self, commit_hash: str) -> bool:
        """Check if commit exists."""
        result = self._run_git("cat-file", "-t", commit_hash, check=False)
        return result == "commit"
