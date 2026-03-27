"""
Tool executor for MCTS-Agent.

Executes various tool calls for code manipulation.
"""

import os
import re
import glob as globlib
import subprocess
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of tool execution."""
    success: bool
    output: str = ""
    error: str = ""
    data: Any = None


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
        self.tools: Dict[str, Callable] = {
            "read": self._read,
            "write": self._write,
            "edit": self._edit,
            "bash": self._bash,
            "grep": self._grep,
            "glob": self._glob,
        }
    
    def execute(self, tool_call: Dict[str, Any]) -> ToolResult:
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
        
        try:
            result = self.tools[tool_name](**args)
            return result
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_name}")
            return ToolResult(
                success=False,
                error=str(e)
            )
    
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
            with open(full_path, "r") as f:
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
            with open(full_path, "r") as f:
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
                    with open(filepath, "r") as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                rel_path = os.path.relpath(filepath, self.root_path)
                                results.append(f"{rel_path}:{line_num}:{line.rstrip()}")
                                
                                if len(results) >= max_results:
                                    break
                except:
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
    
    def list_tools(self) -> List[str]:
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