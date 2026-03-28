"""
Tool executor for ALPHACODE.

Executes various tool calls for code manipulation and information retrieval.
"""

import glob as globlib
import logging
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of tool execution."""
    success: bool
    output: str = ""
    error: str = ""
    data: Any = None


@dataclass
class ToolDefinition:
    """Tool definition for LLM function calling."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


# Tool definitions for LLM function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read file contents with line numbers. Use to examine code or files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Starting line number (0-indexed, optional)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (optional)"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write content to a file. Creates the file if it doesn't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit file by replacing old string with new string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to edit"
                    },
                    "old": {
                        "type": "string",
                        "description": "String to find and replace"
                    },
                    "new": {
                        "type": "string",
                        "description": "New string to replace with"
                    },
                    "all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false)"
                    }
                },
                "required": ["path", "old", "new"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command. Use for running tests, git operations, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Shell command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30)"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a pattern in files using regex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search (default: current)"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Glob pattern for files (default: *)"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.py')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search (default: current)"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
]


class ToolExecutor:
    """
    Tool executor.

    Executes tool calls for:
    - File operations: read, write, edit
    - Search operations: glob, grep
    - Execution: bash
    """

    def __init__(self, root_path: str = None, timeout: int = 30):
        """
        Initialize tool executor.

        Args:
            root_path: Root directory for file operations
            timeout: Default timeout for bash commands
        """
        self.root_path = root_path or os.getcwd()
        self.timeout = timeout

        # Tool registry
        self.tools: dict[str, Callable] = {
            "read": self._read,
            "write": self._write,
            "edit": self._edit,
            "bash": self._bash,
            "grep": self._grep,
            "glob": self._glob,
        }

    def execute(self, tool_call: dict[str, Any]) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: Dict with 'tool' and 'args' keys

        Returns:
            ToolResult
        """
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})

        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}"
            )

        logger.debug(f"Executing tool: {tool_name} with args: {args}")
        try:
            result = self.tools[tool_name](**args)
            logger.debug(f"Tool result: success={result.success}")
            return result
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_name}")
            return ToolResult(
                success=False,
                error=str(e)
            )

    def execute_multiple(self, tool_calls: list[dict[str, Any]]) -> list[ToolResult]:
        """Execute multiple tool calls."""
        return [self.execute(tc) for tc in tool_calls]

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to root."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.root_path, path)

    def _read(
        self,
        path: str,
        offset: int = 0,
        limit: int = None
    ) -> ToolResult:
        """
        Read file contents with line numbers.

        Args:
            path: File path
            offset: Starting line (0-indexed)
            limit: Maximum lines to read
        """
        full_path = self._resolve_path(path)

        if not os.path.exists(full_path):
            return ToolResult(
                success=False,
                error=f"File not found: {path}"
            )

        if os.path.isdir(full_path):
            return ToolResult(
                success=False,
                error=f"Path is a directory: {path}"
            )

        try:
            with open(full_path) as f:
                lines = f.readlines()

            # Apply offset and limit
            end = offset + limit if limit else None
            selected = lines[offset:end]

            # Format with line numbers
            output = "".join(
                f"{offset + i + 1:4}| {line}"
                for i, line in enumerate(selected)
            )

            return ToolResult(
                success=True,
                output=output,
                data={"total_lines": len(lines), "shown_lines": len(selected)}
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _write(self, path: str, content: str) -> ToolResult:
        """
        Write content to file.

        Args:
            path: File path
            content: Content to write
        """
        full_path = self._resolve_path(path)

        try:
            # Create directory if needed
            dir_path = os.path.dirname(full_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            with open(full_path, "w") as f:
                f.write(content)

            logger.info(f"Wrote {len(content)} bytes to {path}")
            return ToolResult(
                success=True,
                output=f"Wrote {len(content)} bytes to {path}"
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _edit(
        self,
        path: str,
        old: str,
        new: str,
        all: bool = False
    ) -> ToolResult:
        """
        Edit file by replacing old string with new.

        Args:
            path: File path
            old: String to replace
            new: Replacement string
            all: Replace all occurrences
        """
        full_path = self._resolve_path(path)

        if not os.path.exists(full_path):
            return ToolResult(
                success=False,
                error=f"File not found: {path}"
            )

        try:
            with open(full_path) as f:
                content = f.read()

            if old not in content:
                return ToolResult(
                    success=False,
                    error=f"String not found in file: {old[:50]}..."
                )

            count = content.count(old)

            if not all and count > 1:
                return ToolResult(
                    success=False,
                    error=f"String appears {count} times, must be unique (use all=true)"
                )

            if all:
                content = content.replace(old, new)
            else:
                content = content.replace(old, new, 1)

            with open(full_path, "w") as f:
                f.write(content)

            return ToolResult(
                success=True,
                output=f"Replaced {count} occurrence(s)"
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _bash(
        self,
        cmd: str,
        timeout: int = None,
        capture_output: bool = True
    ) -> ToolResult:
        """
        Execute bash command.

        Args:
            cmd: Command to execute
            timeout: Timeout in seconds
            capture_output: Whether to capture output
        """
        timeout = timeout or self.timeout

        try:
            # Print command for visibility
            if not capture_output:
                print(f"  │ {cmd}")

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                cwd=self.root_path,
            )

            output = ""
            if capture_output:
                output = result.stdout
                if result.stderr:
                    output += f"\n[stderr]\n{result.stderr}"

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error="" if result.returncode == 0 else f"Exit code: {result.returncode}",
                data={"return_code": result.returncode}
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout}s"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _grep(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        max_results: int = 50
    ) -> ToolResult:
        """
        Search for pattern in files.

        Args:
            pattern: Regex pattern to search
            path: Directory to search
            file_pattern: Glob pattern for files
            max_results: Maximum results to return
        """
        full_path = self._resolve_path(path)

        try:
            regex = re.compile(pattern)
            results = []

            # Find files
            for filepath in globlib.glob(
                os.path.join(full_path, "**", file_pattern),
                recursive=True
            ):
                if not os.path.isfile(filepath):
                    continue

                try:
                    with open(filepath) as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                rel_path = os.path.relpath(filepath, self.root_path)
                                results.append(f"{rel_path}:{line_num}:{line.rstrip()}")

                                if len(results) >= max_results:
                                    break
                except Exception:
                    pass

                if len(results) >= max_results:
                    break

            if not results:
                return ToolResult(
                    success=True,
                    output="No matches found"
                )

            return ToolResult(
                success=True,
                output="\n".join(results),
                data={"count": len(results)}
            )

        except re.error as e:
            return ToolResult(
                success=False,
                error=f"Invalid regex: {e}"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _glob(
        self,
        pattern: str,
        path: str = "."
    ) -> ToolResult:
        """
        Find files matching pattern.

        Args:
            pattern: Glob pattern
            path: Directory to search
        """
        full_path = self._resolve_path(path)

        try:
            files = globlib.glob(
                os.path.join(full_path, pattern),
                recursive=True
            )

            # Make paths relative
            rel_files = [
                os.path.relpath(f, self.root_path)
                for f in files
                if os.path.isfile(f)
            ]

            # Sort by modification time
            rel_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(self.root_path, f)),
                reverse=True
            )

            if not rel_files:
                return ToolResult(
                    success=True,
                    output="No files found"
                )

            return ToolResult(
                success=True,
                output="\n".join(rel_files),
                data={"count": len(rel_files)}
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def register_tool(self, name: str, func: Callable):
        """Register a custom tool."""
        self.tools[name] = func

    def list_tools(self) -> list[str]:
        """List available tools."""
        return list(self.tools.keys())

    def get_tool_description(self, name: str) -> str:
        """Get description of a tool."""
        descriptions = {
            "read": "Read file with line numbers (path, offset?, limit?)",
            "write": "Write content to file (path, content)",
            "edit": "Replace string in file (path, old, new, all?)",
            "bash": "Run shell command (cmd, timeout?)",
            "grep": "Search files for pattern (pattern, path?, file_pattern?)",
            "glob": "Find files matching pattern (pattern, path?)",
        }
        return descriptions.get(name, "No description available")
