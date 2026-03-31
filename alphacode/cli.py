"""
CLI for ALPHACODE.

Interactive command-line interface for code exploration.
LLM autonomously decides when to use MCTS exploration.
"""

import asyncio
import logging
import os
import sys

from alphacode.agent.unified import UnifiedAgent, UnifiedResponse
from alphacode.config import MCTSConfig
from alphacode.core.controller import MCTSController, Solution
from alphacode.llm.client import LLMClient
from alphacode.state.session_manager import SessionManager
from alphacode.tools.executor import ToolExecutor
from alphacode.utils.display import (
    BLUE,
    BOLD,
    CYAN,
    DIM,
    GREEN,
    RED,
    RESET,
    YELLOW,
    Display,
    SYMBOLS,
    format_git_status,
    print_banner,
    print_welcome,
)

# Configure logging - only show warnings and errors by default
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s"
)

# Set specific module log levels
logging.getLogger("alphacode.llm.client").setLevel(logging.WARNING)
logging.getLogger("alphacode.core.controller").setLevel(logging.INFO)
logging.getLogger("alphacode.state.git_manager").setLevel(logging.WARNING)
logging.getLogger("alphacode.state.session_manager").setLevel(logging.WARNING)
logging.getLogger("alphacode.tools.executor").setLevel(logging.WARNING)
logging.getLogger("alphacode.agent.unified").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class MCTSCli:
    """
    ALPHACODE CLI.

    Unified conversation interface with streaming output.
    LLM autonomously decides when to use MCTS exploration.
    """

    def __init__(self, config: MCTSConfig = None):
        self.config = config or MCTSConfig()
        self.controller: MCTSController | None = None
        self.current_solution: Solution | None = None

        # Components
        self.llm_client: LLMClient | None = None
        self.tool_executor: ToolExecutor | None = None
        self.session_manager: SessionManager | None = None

        # Unified agent
        self.agent: UnifiedAgent | None = None

    def _init_components(self):
        """Initialize all components."""
        if self.llm_client is not None:
            return

        if not self.config.llm.api_key:
            return

        # Initialize LLM client
        self.llm_client = LLMClient(
            api_key=self.config.llm.api_key,
            api_base=self.config.llm.api_base,
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            timeout=self.config.llm.timeout,
            max_retries=self.config.llm.max_retries,
            enable_cache=self.config.llm.enable_cache,
            cache_ttl=self.config.llm.cache_ttl,
        )

        # Initialize tools with config
        self.tool_executor = ToolExecutor(
            root_path=os.getcwd(),
            config=self.config,
        )

        # Initialize session manager
        self.session_manager = SessionManager(root_path=os.getcwd())

        # Initialize unified agent
        self.agent = UnifiedAgent(
            llm_client=self.llm_client,
            tool_executor=self.tool_executor,
            session_manager=self.session_manager,
        )

    def _init_agent(self):
        """Initialize agent (backward compatibility)."""
        self._init_components()

    def run(self):
        """Run the CLI."""
        print_banner()
        print_welcome()

        # Show current status
        print(f"{SYMBOLS['folder']} {Display.muted(os.getcwd())}")
        print(f"{SYMBOLS['gear']} {Display.inline_code(self.config.llm.model)}")
        print()

        while True:
            try:
                user_input = input(f"{BOLD}{BLUE}{SYMBOLS['prompt']}{RESET} ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                else:
                    # Process user input with streaming
                    self._process_input_streaming(user_input)

            except (KeyboardInterrupt, EOFError):
                print(f"\n{GREEN}Goodbye!{RESET}")
                break
            except Exception as e:
                print(f"{RED}Error: {e}{RESET}")
                import traceback
                traceback.print_exc()

    def _process_input_streaming(self, user_input: str):
        """
        Process user input with streaming output.

        Real-time token display for better UX.
        """
        # Initialize components if needed
        self._init_components()

        # Check if agent is available
        if not self.agent:
            print(f"{RED}Error: No LLM client available.{RESET}")
            print("Set NVIDIA_API_KEY or OPENAI_API_KEY environment variable.")
            return

        print()  # Blank line

        async def stream_process():
            tool_calls_info = []
            full_content = ""

            def on_token(token):
                # Print token immediately
                print(token, end="", flush=True)

            def on_tool_call(tc):
                tool_calls_info.append(tc)
                # Only show tool name, not args (args might be long JSON)
                print(f"{DIM}🔧 {tc['tool']}...{RESET}", flush=True)

            def on_tool_result(result):
                # Show tool execution result clearly
                print()  # newline before result
                print(f"{CYAN}{result}{RESET}")
                print()  # newline after result

            def on_mcts_progress(progress):
                # Show MCTS progress briefly
                if progress.get('improved'):
                    print(f"{GREEN}↑ score: {progress['score']:.2f}{RESET}", flush=True)

            # Stream the response
            async for token in self.agent.process_stream(
                user_input,
                on_token=on_token,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                on_mcts_progress=on_mcts_progress,
            ):
                full_content += token

            # Final newline
            print()

        asyncio.run(stream_process())

    def _handle_command(self, cmd: str):
        """Handle slash commands."""
        cmd_lower = cmd.lower().strip()

        if cmd_lower in ["/help", "/h", "/?"]:
            self._show_help()
        elif cmd_lower in ["/status", "/s"]:
            self._show_status()
        elif cmd_lower in ["/tree", "/t"]:
            self._show_tree()
        elif cmd_lower in ["/best", "/b"]:
            self._show_best()
        elif cmd_lower in ["/alternatives", "/alt"]:
            self._show_alternatives()
        elif cmd_lower in ["/report", "/r"]:
            self._show_report()
        elif cmd_lower == "/config":
            self._show_config()
        elif cmd_lower == "/adopt":
            self._adopt_solution()
        elif cmd_lower in ["/continue", "/c"]:
            self._continue_exploration()
        elif cmd_lower == "/clear":
            self._clear_history()
        elif cmd_lower in ["/quit", "/q"]:
            raise KeyboardInterrupt()
        else:
            print(f"{YELLOW}Unknown command: {cmd}{RESET}")
            print(f"Type {BOLD}/help{RESET} for available commands.")

    def _show_help(self):
        """Show help message."""
        help_text = f"""
{BOLD}ALPHACODE - Unified Agent with Tools{RESET}

{BOLD}Commands:{RESET}
  /help, /h             Show this help message
  /status, /s           Show current exploration status
  /tree, /t             Show search tree visualization
  /best, /b             Show best solution found
  /alternatives, /alt   Show alternative solutions
  /report, /r           Generate full exploration report
  /config               Show current configuration
  /adopt                Adopt best solution (merge to main)
  /continue, /c         Continue exploration
  /clear                Clear conversation history
  /quit, /q             Exit ALPHACODE

{BOLD}Tools:{RESET}
  read, write, edit    - File operations
  grep, glob           - Search operations
  bash                 - Shell commands
  mcts_explore         - Code exploration

{BOLD}Usage:{RESET}
  Just type your message. The agent will decide how to help you.
"""
        print(help_text)

    def _show_status(self):
        """Show exploration status."""
        if not self.controller or not self.controller.search_tree:
            print(f"{YELLOW}No active MCTS session.{RESET}")
            print("Start a code task to begin exploration.")
            return

        tree = self.controller.search_tree
        stats = tree.get_stats()

        print(f"\n{BOLD}Session:{RESET} {tree.session_id}")
        print(f"{BOLD}Goal:{RESET} {tree.goal}")
        print(f"{BOLD}Status:{RESET} {'Running' if self.controller.running else 'Stopped'}")
        print()

        # Progress bar
        progress = Display.progress_bar(
            self.controller.iteration,
            self.config.max_iterations,
            label="Progress"
        )
        print(progress)
        print()

        # Statistics
        print(f"{BOLD}Statistics:{RESET}")
        print(f"  Nodes: {stats['total_nodes']}")
        print(f"  Depth: {stats['max_depth']}")
        print(f"  Best score: {stats['best_value']:.3f}")
        print(f"  Grid coverage: {stats['grid_coverage']:.1%}")
        print()

        # Islands
        print(f"{BOLD}Islands:{RESET}")
        for island_stats in stats["islands"]:
            print(
                f"  Island {island_stats['id']}: "
                f"{island_stats['size']} nodes, "
                f"best={island_stats['best_value']:.3f}"
            )

    def _show_tree(self):
        """Show search tree."""
        if not self.controller:
            print(f"{YELLOW}No active session.{RESET}")
            return

        # Use Git visualization
        viz = self.controller.git_manager.get_tree_visualization()

        print(f"\n{BOLD}Search Tree (Git branches):{RESET}")
        print(f"{DIM}{viz}{RESET}")

    def _show_best(self):
        """Show best solution."""
        if not self.current_solution:
            print(f"{YELLOW}No solution found yet.{RESET}")
            return

        best = self.current_solution.best_node

        if not best:
            print(f"{YELLOW}No best solution available.{RESET}")
            return

        print(f"\n{BOLD}Best Solution:{RESET}")
        print(f"{BOLD}Score:{RESET} {best.value:.3f}")
        print(f"{BOLD}Path:{RESET} {best.git_branch or 'default'}")
        print()

        # Show code
        print(f"{BOLD}Code:{RESET}")
        print(Display.code(best.code))

    def _show_alternatives(self):
        """Show alternative solutions."""
        if not self.current_solution:
            print(f"{YELLOW}No solutions found yet.{RESET}")
            return

        alternatives = self.current_solution.alternatives

        if not alternatives:
            print(f"{YELLOW}No alternatives available.{RESET}")
            return

        print(f"\n{BOLD}Alternative Solutions:{RESET}")

        for i, alt in enumerate(alternatives, 1):
            print(f"\n{BOLD}Alternative {i}:{RESET}")
            print(f"  Score: {alt['score']:.3f}")
            print(f"  Branch: {alt['branch']}")
            print()
            print(Display.code(alt['code'], max_lines=20))

    def _show_report(self):
        """Show full exploration report."""
        if not self.controller:
            print(f"{YELLOW}No active session.{RESET}")
            return

        report = self.controller.generate_report()

        print(f"\n{BOLD}Exploration Report:{RESET}")
        print(report)

    def _show_config(self):
        """Show current configuration."""
        print(f"\n{BOLD}Configuration:{RESET}")
        print(f"  Model: {self.config.llm.model}")
        print(f"  Temperature: {self.config.llm.temperature}")
        print(f"  Max tokens: {self.config.llm.max_tokens}")
        print(f"  Max iterations: {self.config.max_iterations}")
        print(f"  Exploration weight: {self.config.exploration_weight}")
        print(f"  Cache enabled: {self.config.llm.enable_cache}")

    def _adopt_solution(self):
        """Adopt best solution."""
        if not self.current_solution:
            print(f"{YELLOW}No solution to adopt.{RESET}")
            return

        best = self.current_solution.best_node

        if not best:
            print(f"{YELLOW}No best solution available.{RESET}")
            return

        # Merge to main
        self.controller.git_manager.adopt_solution(best.git_branch)

        print(f"{GREEN}✓ Adopted solution from branch: {best.git_branch}{RESET}")
        print(f"  Score: {best.value:.3f}")

    def _continue_exploration(self):
        """Continue exploration."""
        if not self.controller:
            print(f"{YELLOW}No active session.{RESET}")
            return

        print(f"{CYAN}Continuing exploration...{RESET}")
        self.controller.run_iteration()

    def _clear_history(self):
        """Clear conversation history."""
        if self.agent:
            self.agent.clear_history()
        print(f"{GREEN}✓ Conversation history cleared.{RESET}")


def main():
    """Main entry point."""
    config = MCTSConfig()
    cli = MCTSCli(config)
    cli.run()


if __name__ == "__main__":
    main()