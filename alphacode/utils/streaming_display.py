"""
Streaming display utilities for real-time UI updates.

Provides visual feedback during long-running operations.
All decisions remain with the LLM - this only improves presentation.
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"


@dataclass
class StreamState:
    """Current streaming state."""
    text: str = ""
    tokens: int = 0
    start_time: float = field(default_factory=time.time)
    is_thinking: bool = False
    is_code: bool = False
    code_block_start: int = -1


class StreamingDisplay:
    """
    Real-time display for LLM streaming output.
    
    Design principle: This only handles presentation.
    All content decisions come from the LLM.
    """

    def __init__(self, show_speed: bool = True, show_tokens: bool = True):
        """
        Initialize streaming display.
        
        Args:
            show_speed: Show tokens/second
            show_tokens: Show token count
        """
        self.show_speed = show_speed
        self.show_tokens = show_tokens
        self.state = StreamState()
        self._last_line_length = 0

    def _get_elapsed(self) -> float:
        """Get elapsed time."""
        return time.time() - self.state.start_time

    def _get_speed(self) -> float:
        """Calculate tokens per second."""
        elapsed = self._get_elapsed()
        if elapsed > 0 and self.state.tokens > 0:
            return self.state.tokens / elapsed
        return 0.0

    def _clear_status_line(self):
        """Clear the status line."""
        sys.stderr.write("\r\033[K")  # Clear line
        sys.stderr.flush()

    def _show_status(self, message: str = ""):
        """Show status line at bottom."""
        self._clear_status_line()
        
        parts = []
        
        # Token count
        if self.show_tokens and self.state.tokens > 0:
            parts.append(f"{DIM}tokens: {self.state.tokens}{RESET}")
        
        # Speed
        if self.show_speed and self.state.tokens > 0:
            speed = self._get_speed()
            parts.append(f"{DIM}speed: {speed:.1f} t/s{RESET}")
        
        # Time
        elapsed = self._get_elapsed()
        parts.append(f"{DIM}time: {elapsed:.1f}s{RESET}")
        
        status = "  ".join(parts)
        if status:
            sys.stderr.write(f"\n{DIM}[{status}]{RESET}\r")
            sys.stderr.flush()

    def on_token(self, token: str):
        """
        Handle a new token from the stream.
        
        Args:
            token: New token from LLM
        """
        self.state.tokens += 1
        self.state.text += token
        
        # Detect code blocks
        if token == "`" and self.state.text.rstrip().endswith("```"):
            self.state.is_code = not self.state.is_code
        
        # Print the token
        print(token, end="", flush=True)
        
        # Update status periodically (every 10 tokens)
        if self.state.tokens % 10 == 0:
            self._show_status()

    def on_complete(self) -> str:
        """
        Handle stream completion.
        
        Returns:
            Complete text
        """
        self._clear_status_line()
        
        # Print final stats
        if self.show_tokens or self.show_speed:
            elapsed = self._get_elapsed()
            speed = self._get_speed()
            print(f"\n{DIM}[{self.state.tokens} tokens, {speed:.1f} t/s, {elapsed:.1f}s]{RESET}")
        
        return self.state.text

    def reset(self):
        """Reset for new stream."""
        self.state = StreamState()


class MCTSProgressDisplay:
    """
    Display MCTS exploration progress in real-time.
    
    Shows what the LLM is exploring without making decisions.
    """

    def __init__(self, total_iterations: int = 20):
        """
        Initialize MCTS progress display.
        
        Args:
            total_iterations: Expected total iterations
        """
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.best_score = 0.0
        self.nodes_explored = 0
        self._start_time = time.time()

    def _clear_line(self):
        """Clear current line."""
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    def show_thinking(self, message: str = "Exploring solutions"):
        """Show thinking indicator."""
        elapsed = time.time() - self._start_time
        print(f"{DIM}🧠 {message}... ({elapsed:.1f}s){RESET}")

    def update(
        self,
        iteration: int = None,
        best_score: float = None,
        nodes_explored: int = None,
        message: str = None
    ):
        """
        Update progress display.
        
        Args:
            iteration: Current iteration
            best_score: Best score so far
            nodes_explored: Number of nodes explored
            message: Optional message
        """
        if iteration is not None:
            self.current_iteration = iteration
        if best_score is not None:
            self.best_score = best_score
        if nodes_explored is not None:
            self.nodes_explored = nodes_explored

        # Build progress bar
        ratio = self.current_iteration / self.total_iterations if self.total_iterations > 0 else 0
        bar_width = 20
        filled = int(bar_width * ratio)
        bar = f"{GREEN}{'█' * filled}{DIM}{'░' * (bar_width - filled)}{RESET}"
        percentage = int(ratio * 100)

        # Score color
        if self.best_score >= 0.8:
            score_color = GREEN
        elif self.best_score >= 0.6:
            score_color = YELLOW
        else:
            score_color = CYAN

        # Build status line
        status = (
            f"\r{DIM}Iter:{RESET} {self.current_iteration}/{self.total_iterations} "
            f"{bar} {percentage}%  "
            f"{DIM}Score:{RESET} {score_color}{self.best_score:.2f}{RESET}  "
            f"{DIM}Nodes:{RESET} {self.nodes_explored}"
        )
        
        if message:
            status += f"  {DIM}{message}{RESET}"

        self._clear_line()
        sys.stderr.write(status)
        sys.stderr.flush()

    def show_best_code(self, code: str, score: float, max_lines: int = 5):
        """
        Show preview of best code found so far.
        
        Args:
            code: Best code
            score: Code score
            max_lines: Maximum lines to show
        """
        self._clear_line()
        
        print(f"\n{GREEN}✨ Better solution found (score: {score:.2f}){RESET}")
        
        lines = code.strip().split("\n")[:max_lines]
        for i, line in enumerate(lines, 1):
            print(f"{DIM}{i:3}│{RESET} {line}")
        
        if len(code.split("\n")) > max_lines:
            remaining = len(code.split("\n")) - max_lines
            print(f"{DIM}   │ ... ({remaining} more lines){RESET}")
        
        print()

    def complete(self, final_score: float, total_nodes: int):
        """
        Show completion message.
        
        Args:
            final_score: Final best score
            total_nodes: Total nodes explored
        """
        self._clear_line()
        elapsed = time.time() - self._start_time
        
        if final_score >= 0.8:
            icon = "✨"
            color = GREEN
        elif final_score >= 0.6:
            icon = "✓"
            color = YELLOW
        else:
            icon = "⚠"
            color = CYAN

        print(f"\n{color}{icon} Exploration complete!{RESET}")
        print(f"   Best score: {color}{final_score:.2f}{RESET}")
        print(f"   Nodes explored: {total_nodes}")
        print(f"   Time: {elapsed:.1f}s")


async def stream_and_display(
    stream_generator: AsyncIterator[str],
    display: StreamingDisplay = None,
) -> str:
    """
    Stream LLM output and display in real-time.
    
    Args:
        stream_generator: Async generator yielding tokens
        display: Optional display instance
        
    Returns:
        Complete text
    """
    if display is None:
        display = StreamingDisplay()
    
    display.reset()
    
    async for token in stream_generator:
        display.on_token(token)
        # Allow other tasks to run
        await asyncio.sleep(0)
    
    return display.on_complete()


def format_speed_info(tokens: int, elapsed: float) -> str:
    """
    Format speed information.
    
    Args:
        tokens: Number of tokens
        elapsed: Elapsed time in seconds
        
    Returns:
        Formatted string
    """
    if elapsed > 0 and tokens > 0:
        speed = tokens / elapsed
        return f"{tokens} tokens, {speed:.1f} t/s, {elapsed:.1f}s"
    return f"{tokens} tokens, {elapsed:.1f}s"