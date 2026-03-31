"""
Display utilities for MCTS-Agent CLI.

Human-friendly UI design principles:
- Clear visual hierarchy
- Consistent color coding
- Meaningful icons and symbols
- Progress indicators
- Compact but readable layout
"""

import os
import shutil

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"

# Foreground colors
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

# Bright colors
BRIGHT_BLACK = "\033[90m"
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"

# Background colors
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"
BG_CYAN = "\033[46m"

# Symbols
SYMBOLS = {
    "check": "✓",
    "cross": "✗",
    "arrow": "→",
    "bullet": "•",
    "star": "★",
    "heart": "♥",
    "info": "ℹ",
    "warning": "⚠",
    "question": "?",
    "prompt": "❯",
    "branch": "⎇",
    "commit": "○",
    "folder": "📁",
    "file": "📄",
    "gear": "⚙",
    "sparkle": "✨",
    "rocket": "🚀",
    "lightbulb": "💡",
    "search": "🔍",
    "tools": "🛠",
    "chat": "💬",
    "code": "📝",
    "success": "✅",
    "error": "❌",
    "warning_emoji": "⚠️",
    "info_emoji": "ℹ️",
}


def get_terminal_width() -> int:
    """Get terminal width, fallback to 80."""
    try:
        return shutil.get_terminal_size().columns
    except (OSError, AttributeError):
        return 80


def truncate_text(text: str, max_width: int, suffix: str = "...") -> str:
    """Truncate text to fit within max_width."""
    if len(text) <= max_width:
        return text
    return text[:max_width - len(suffix)] + suffix


class Display:
    """Display utilities for CLI."""

    @staticmethod
    def separator(char: str = "─", width: int = None) -> str:
        """Create a separator line."""
        if width is None:
            width = min(get_terminal_width(), 80)
        return f"{DIM}{char * width}{RESET}"

    @staticmethod
    def header(text: str, icon: str = "⏺") -> str:
        """Create a header with icon."""
        return f"\n{BOLD}{CYAN}{icon} {text}{RESET}"

    @staticmethod
    def section(title: str, icon: str = "•") -> str:
        """Create a section header."""
        return f"\n{BOLD}{BLUE}{icon} {title}{RESET}"

    @staticmethod
    def subsection(title: str) -> str:
        """Create a subsection header."""
        return f"{BOLD}{WHITE}{title}{RESET}"

    @staticmethod
    def success(text: str, icon: str = "✓") -> str:
        """Format success message."""
        return f"{GREEN}{icon} {text}{RESET}"

    @staticmethod
    def error(text: str, icon: str = "✗") -> str:
        """Format error message."""
        return f"{RED}{icon} {text}{RESET}"

    @staticmethod
    def warning(text: str, icon: str = "⚠") -> str:
        """Format warning message."""
        return f"{YELLOW}{icon} {text}{RESET}"

    @staticmethod
    def info(text: str, icon: str = "ℹ") -> str:
        """Format info message."""
        return f"{CYAN}{icon} {text}{RESET}"

    @staticmethod
    def dim(text: str) -> str:
        """Dim text."""
        return f"{DIM}{text}{RESET}"

    @staticmethod
    def bold(text: str) -> str:
        """Bold text."""
        return f"{BOLD}{text}{RESET}"

    @staticmethod
    def muted(text: str) -> str:
        """Muted/secondary text."""
        return f"{BRIGHT_BLACK}{text}{RESET}"

    @staticmethod
    def code(code: str, language: str = "python", max_lines: int = 20) -> str:
        """Format code block with smart truncation."""
        lines = code.split("\n")
        total_lines = len(lines)
        
        if total_lines > max_lines:
            shown = lines[:max_lines]
            code = "\n".join(shown)
            truncated_info = f"\n{DIM}... ({total_lines - max_lines} more lines){RESET}"
        else:
            truncated_info = ""
        
        # Add line numbers
        numbered = "\n".join(
            f"{DIM}{i+1:3}│{RESET} {line}"
            for i, line in enumerate(lines[:max_lines])
        )
        
        return f"{DIM}```{language}{RESET}\n{numbered}{truncated_info}\n{DIM}```{RESET}"

    @staticmethod
    def inline_code(code: str) -> str:
        """Format inline code."""
        return f"{DIM}`{code}`{RESET}"

    @staticmethod
    def progress_bar(
        current: int,
        total: int,
        width: int = 20,
        label: str = "",
        show_percent: bool = True
    ) -> str:
        """Create a progress bar."""
        if total == 0:
            ratio = 1.0
        else:
            ratio = min(current / total, 1.0)

        filled = int(width * ratio)
        empty = width - filled

        bar = f"{GREEN}{'█' * filled}{RESET}{DIM}{'░' * empty}{RESET}"

        result = ""
        if label:
            result = f"{label} {bar}"
        else:
            result = bar
        
        if show_percent:
            result += f" {int(ratio * 100)}%"
        
        result += f" ({current}/{total})"
        
        return result

    @staticmethod
    def spinner(step: int) -> str:
        """Get spinner character for loading state."""
        spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        return spinners[step % len(spinners)]

    @staticmethod
    def status(text: str, state: str = "info") -> str:
        """Create a status indicator."""
        icons = {
            "info": ("ℹ", CYAN),
            "success": ("✓", GREEN),
            "error": ("✗", RED),
            "warning": ("⚠", YELLOW),
            "working": ("◉", BLUE),
            "pending": ("○", DIM),
        }
        icon, color = icons.get(state, ("•", WHITE))
        return f"{color}{icon}{RESET} {text}"

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
            f"{BOLD}{str(h):<{widths[i]}}{RESET}"
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
            f"{DIM}{separator}{RESET}",
            *row_lines
        ])

    @staticmethod
    def tree(nodes: list, indent: str = "") -> str:
        """Create a tree visualization."""
        lines = []

        for i, node in enumerate(nodes):
            is_last = i == len(nodes) - 1

            prefix = indent + ("└── " if is_last else "├── ")
            lines.append(f"{prefix}{node}")

        return "\n".join(lines)

    @staticmethod
    def metric(name: str, value: str, unit: str = "") -> str:
        """Format a metric display."""
        unit_str = f" {unit}" if unit else ""
        return f"{DIM}{name}:{RESET} {BOLD}{value}{unit_str}{RESET}"

    @staticmethod
    def key_value(key: str, value: str, key_width: int = 15) -> str:
        """Format a key-value pair."""
        return f"{DIM}{key:>{key_width}}{RESET} {value}"

    @staticmethod
    def list_item(text: str, level: int = 0, bullet: str = "•") -> str:
        """Format a list item with indentation."""
        indent = "  " * level
        return f"{indent}{DIM}{bullet}{RESET} {text}"

    @staticmethod
    def tag(text: str, color: str = BLUE) -> str:
        """Format a tag/label."""
        return f"{color}[{text}]{RESET}"

    @staticmethod
    def path(text: str) -> str:
        """Format a file path."""
        return f"{CYAN}{text}{RESET}"

    @staticmethod
    def timestamp(text: str) -> str:
        """Format a timestamp."""
        return f"{DIM}{text}{RESET}"


def print_banner():
    """Print ALPHACODE banner with ASCII art."""
    width = get_terminal_width()
    
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


def print_welcome():
    """Print welcome message."""
    print(f"""
{BOLD}Welcome to ALPHACODE!{RESET} {SYMBOLS['sparkle']}

{SYMBOLS['chat']} {CYAN}Chat with me:{RESET} Type your message and I'll help you.

{Display.muted("Type /help for commands, or just start typing...")}
""")


def format_file_list(files: list[str], max_display: int = 10) -> str:
    """Format a list of files for display."""
    if not files:
        return Display.muted("(no files)")
    
    lines = []
    for f in files[:max_display]:
        icon = "📁" if f.endswith("/") else "📄"
        lines.append(f"  {icon} {Display.path(f)}")
    
    if len(files) > max_display:
        lines.append(f"  {Display.muted(f'... and {len(files) - max_display} more')}")
    
    return "\n".join(lines)


def format_git_status(branch: str, is_clean: bool, modified: int = 0, untracked: int = 0) -> str:
    """Format Git status for display."""
    status_icon = SYMBOLS["check"] if is_clean else SYMBOLS["warning"]
    status_color = GREEN if is_clean else YELLOW
    
    parts = [f"{SYMBOLS['branch']} {Display.inline_code(branch)}"]
    
    if not is_clean:
        if modified:
            parts.append(f"{YELLOW}{modified} modified{RESET}")
        if untracked:
            parts.append(f"{CYAN}{untracked} untracked{RESET}")
    else:
        parts.append(f"{GREEN}clean{RESET}")
    
    return " | ".join(parts)