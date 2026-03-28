"""
Display utilities for MCTS-Agent CLI.
"""

import os

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Colors
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

# Background colors
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"


class Display:
    """Display utilities for CLI."""

    @staticmethod
    def separator(char: str = "─", width: int = None) -> str:
        """Create a separator line."""
        if width is None:
            try:
                width = min(os.get_terminal_size().columns, 80)
            except OSError:
                width = 80  # Default width when not in terminal
        return f"{DIM}{char * width}{RESET}"

    @staticmethod
    def header(text: str) -> str:
        """Create a header."""
        return f"\n{BOLD}{CYAN}⏺ {text}{RESET}"

    @staticmethod
    def success(text: str) -> str:
        """Format success message."""
        return f"{GREEN}✓ {text}{RESET}"

    @staticmethod
    def error(text: str) -> str:
        """Format error message."""
        return f"{RED}✗ {text}{RESET}"

    @staticmethod
    def warning(text: str) -> str:
        """Format warning message."""
        return f"{YELLOW}⚠ {text}{RESET}"

    @staticmethod
    def info(text: str) -> str:
        """Format info message."""
        return f"{CYAN}ℹ {text}{RESET}"

    @staticmethod
    def code(code: str, language: str = "python") -> str:
        """Format code block."""
        lines = code.split("\n")
        if len(lines) > 20:
            code = "\n".join(lines[:20]) + f"\n{DIM}... ({len(lines) - 20} more lines){RESET}"

        return f"{DIM}```{language}{RESET}\n{code}\n{DIM}```{RESET}"

    @staticmethod
    def progress_bar(
        current: int,
        total: int,
        width: int = 20,
        label: str = ""
    ) -> str:
        """Create a progress bar."""
        if total == 0:
            ratio = 1.0
        else:
            ratio = current / total

        filled = int(width * ratio)
        empty = width - filled

        bar = f"[{GREEN}{'█' * filled}{RESET}{'░' * empty}]"

        if label:
            return f"{label} {bar} {current}/{total}"
        return f"{bar} {current}/{total}"

    @staticmethod
    def table(headers: list, rows: list, widths: list = None) -> str:
        """Create a simple table."""
        if not rows:
            return ""

        if widths is None:
            # Auto-calculate widths
            widths = [
                max(len(str(row[i])) if i < len(row) else 0 for row in [headers] + rows)
                for i in range(len(headers))
            ]

        # Build header
        header_line = " | ".join(
            f"{str(h):<{widths[i]}}"
            for i, h in enumerate(headers)
        )

        separator = "-+-".join(
            "-" * w
            for w in widths
        )

        # Build rows
        row_lines = []
        for row in rows:
            row_line = " | ".join(
                f"{str(row[i]):<{widths[i]}}" if i < len(row) else ""
                for i in range(len(headers))
            )
            row_lines.append(row_line)

        return "\n".join([
            header_line,
            separator,
            *row_lines
        ])

    @staticmethod
    def tree(nodes: list, indent: str = "") -> str:
        """Create a tree visualization."""
        lines = []

        for i, node in enumerate(nodes):
            is_last = i == len(nodes) - 1

            prefix = indent + ("└── " if is_last else "├── ")
            indent + ("    " if is_last else "│   ")

            lines.append(f"{prefix}{node}")

        return "\n".join(lines)


def print_banner():
    """Print ALPHACODE banner with ASCII art."""
    # Check terminal width
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80

    # Full ASCII art requires ~80 columns
    if width >= 80:
        banner = f"""
{CYAN}      .o.       oooo             oooo                    .oooooo.                   .o8
{CYAN}     .888.      `888             `888                   d8P'  `Y8b                 "888
{CYAN}    .8"888.      888  oo.ooooo.   888 .oo.    .oooo.   888           .ooooo.   .oooo888   .ooooo.
{CYAN}   .8' `888.     888   888' `88b  888P"Y88b  `P  )88b  888          d88' `88b d88' `888  d88' `88b
{CYAN}  .88ooo8888.    888   888   888  888   888   .oP"888  888          888   888 888   888  888ooo888
{CYAN} .8'     `888.   888   888   888  888   888  d8(  888  `88b    ooo  888   888 888   888  888    .o
{CYAN}o88o     o8888o o888o  888bod8P' o888o o888o `Y888""8o  `Y8bood8P'  `Y8bod8P' `Y8bod88P" `Y8bod8P'
{CYAN}                       888
{CYAN}                      o888o                                                                        {RESET}
{DIM}       ALPHACODE - MCTS-based Code Exploration with Git  v0.1.0{RESET}
"""
    else:
        # Simplified for narrow terminals
        banner = f"""
{BOLD}{CYAN}╔═════════════════════════════════╗
║      🧠 ALPHACODE v0.1.0        ║
║  MCTS Code Exploration with Git ║
╚═════════════════════════════════╝{RESET}
"""
    print(banner)
