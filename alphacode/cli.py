"""
CLI for ALPHACODE.

Interactive command-line interface for code exploration.
"""

import logging
import os

from alphacode.config import MCTSConfig
from alphacode.core.agent import Agent, AgentResponse
from alphacode.core.controller import MCTSController, Solution
from alphacode.llm.client import LLMClient
from alphacode.llm.intent import IntentDetector, IntentType
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
    print_banner,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MCTSCli:
    """
    ALPHACODE CLI.

    Interactive interface for code exploration.
    All intents can use tools (controlled by LLM).
    Only CODE_TASK goes through MCTS exploration for optimization.
    """

    def __init__(self, config: MCTSConfig = None):
        self.config = config or MCTSConfig()
        self.controller: MCTSController | None = None
        self.current_solution: Solution | None = None

        # Unified agent for all intents (with tool support)
        self.llm_client: LLMClient | None = None
        self.tool_executor: ToolExecutor | None = None
        self.agent: Agent | None = None
        self.intent_detector: IntentDetector | None = None

    def _init_agent(self):
        """Initialize the unified agent with tools."""
        if self.agent is None and self.config.llm.api_key:
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
            self.tool_executor = ToolExecutor(root_path=os.getcwd())
            self.agent = Agent(
                llm_client=self.llm_client,
                tool_executor=self.tool_executor,
            )
            self.intent_detector = IntentDetector(self.llm_client)

    def run(self):
        """Run the CLI."""
        print_banner()

        print(f"{BOLD}Working Directory:{RESET} {os.getcwd()}")
        print(f"{BOLD}Model:{RESET} {self.config.llm.model}")
        print()
        print(f"Type {BOLD}help{RESET} for commands, or start chatting with me.")
        print()

        while True:
            try:
                user_input = input(f"{BOLD}{BLUE}❯{RESET} ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                else:
                    # Process user input with unified agent
                    self._process_input(user_input)

            except (KeyboardInterrupt, EOFError):
                print(f"\n{GREEN}Goodbye!{RESET}")
                break
            except Exception as e:
                print(f"{RED}Error: {e}{RESET}")
                import traceback
                traceback.print_exc()

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
        elif cmd_lower in ["/alternatives", "/alt", "/a"]:
            self._show_alternatives()
        elif cmd_lower in ["/adopt"]:
            self._adopt_best()
        elif cmd_lower in ["/continue", "/c"]:
            self._continue_exploration()
        elif cmd_lower in ["/report", "/r"]:
            self._show_report()
        elif cmd_lower in ["/clear"]:
            self._clear_conversation()
        elif cmd_lower in ["/config"]:
            self._show_config()
        elif cmd_lower in ["/quit", "/q", "/exit"]:
            raise EOFError
        else:
            print(f"{YELLOW}Unknown command: {cmd}{RESET}")
            print(f"Type {BOLD}/help{RESET} for available commands.")

    def _process_input(self, user_input: str):
        """
        Process user input with unified agent.

        All intents can use tools (LLM decides).
        Only CODE_TASK goes to MCTS exploration after initial generation.
        """
        # Initialize agent if needed
        self._init_agent()

        # Check if agent is available
        if not self.agent:
            print(f"{RED}Error: No LLM client available.{RESET}")
            print("Set NVIDIA_API_KEY or OPENAI_API_KEY environment variable.")
            return

        # Detect intent
        print(f"\n{DIM}Analyzing your request...{RESET}")

        if self.intent_detector:
            intent_result = self.intent_detector.detect_sync(user_input)
        else:
            from alphacode.llm.intent import IntentResult
            intent_result = IntentResult(
                intent=IntentType.CODE_TASK,
                confidence=0.5,
                reason="No LLM for intent detection",
                code_hint=user_input,
            )

        intent_msg = (
            f"{DIM}Intent: {intent_result.intent.value} "
            f"(confidence: {intent_result.confidence:.2f}){RESET}"
        )
        print(intent_msg)

        # For CODE_TASK, we use MCTS for exploration
        # For other intents, use agent with tools
        if intent_result.intent == IntentType.CODE_TASK:
            # Let agent do initial code generation with tools
            print(f"\n{DIM}Generating initial solution...{RESET}")
            agent_response = self.agent.process_sync(
                user_input,
                system_prompt=Agent.CODE_SYSTEM,
            )

            # Show what agent did
            self._show_agent_response(agent_response)

            # Now run MCTS exploration for optimization
            goal = intent_result.code_hint or user_input
            self._run_mcts_exploration(goal)

        else:
            # For questions/chitchat - use agent with tools
            print(f"\n{DIM}Processing with tools...{RESET}")
            agent_response = self.agent.process_sync(user_input)

            # Show response
            self._show_agent_response(agent_response)

    def _show_agent_response(self, response: AgentResponse):
        """Display agent response including tool usage."""
        # Show tool results directly (since LLM may not format them well)
        if response.tool_results:
            print(f"\n{DIM}Tool results:{RESET}")
            for tr in response.tool_results:
                if tr.get("success"):
                    # Show the actual tool output
                    print(tr["result"])
                else:
                    print(f"{RED}Error: {tr.get('result')}{RESET}")

        # Show LLM content if any
        if response.content and not response.tool_results:
            print(f"\n{CYAN}{response.content}{RESET}\n")
        elif response.content:
            # LLM summary after tool results
            print(f"\n{DIM}{response.content}{RESET}\n")

    def _run_mcts_exploration(self, goal: str):
        """Run MCTS exploration for code optimization."""
        print(f"\n{Display.separator()}")
        print(f"{CYAN}MCTS Exploration:{RESET} {goal}")
        print(f"{Display.separator()}\n")

        # Check for existing code
        initial_code = ""
        program_file = "program.py"

        if os.path.exists(program_file):
            print(f"{DIM}Found existing {program_file}, using as starting point{RESET}")
            with open(program_file) as f:
                initial_code = f.read()

        # Create controller and run MCTS
        self.controller = MCTSController(self.config)

        start_msg = f"{DIM}Starting exploration "
        start_msg += f"(max {self.config.max_iterations} iterations)...{RESET}\n"
        print(start_msg)

        solution = self.controller.solve(goal, initial_code)
        self.current_solution = solution

        # Display result
        print(f"\n{Display.separator()}")

        if solution.test_passed:
            print(Display.header(f"Solution Found & Tested (score: {solution.best_score:.3f})"))
        else:
            print(Display.header(f"Solution Found (score: {solution.best_score:.3f})"))

        print(f"{Display.separator()}\n")

        print(f"{BOLD}Iterations:{RESET} {solution.iterations}")
        print(f"{BOLD}Nodes explored:{RESET} {solution.total_nodes}")
        test_status = GREEN if solution.test_passed else RED
        print(f"{BOLD}Test passed:{RESET} {test_status}{solution.test_passed}{RESET}")

        print(f"\n{BOLD}Code:{RESET}")
        print(Display.code(solution.best_code))

        if not solution.test_passed and solution.test_output:
            print(f"\n{YELLOW}Test output:{RESET}")
            print(f"{DIM}{solution.test_output[:300]}{RESET}")

        # Suggest next steps
        alternatives = solution.get_alternative_solutions(3)
        if alternatives:
            print(f"\n{YELLOW}{len(alternatives)} alternative solutions available.{RESET}")
            print(f"Type {BOLD}/alternatives{RESET} to view them.")

        print(f"\nType {BOLD}/adopt{RESET} to merge best solution to main branch.")

    def _show_help(self):
        """Show help message."""
        help_text = f"""
{BOLD}ALPHACODE - Unified Agent with Tools{RESET}

{BOLD}Chat & Tools:{RESET}
  <any message>         Chat with me - I can use tools to help you
                        - Questions: I'll search/read files to answer
                        - Code tasks: I'll write code, then optimize with MCTS
                        - Chitchat: I'll just chat with you

{BOLD}Information:{RESET}
  /help, /h             Show this help message
  /status, /s           Show current exploration status
  /tree, /t             Show search tree visualization
  /best, /b             Show best solution found
  /alternatives, /alt   Show alternative solutions
  /report, /r           Generate full exploration report
  /config               Show current configuration

{BOLD}Actions:{RESET}
  /adopt                Adopt best solution (merge to main)
  /continue, /c         Continue exploration
  /clear                Clear conversation history
  /quit, /q             Exit ALPHACODE

{BOLD}Examples:{RESET}
  ❯ What files are in this project?     (I'll use glob tool)
  ❯ Read main.py and explain it         (I'll use read tool)
  ❯ Write a binary search tree          (I'll write code, then MCTS optimize)
  ❯ What is quick sort?                 (I'll explain, may show code)
  ❯ Hello!                              (Just chatting)

{BOLD}Available Tools:{RESET}
  read, write, edit    - File operations
  grep, glob           - Search operations
  bash                 - Run shell commands
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

        print(f"\n{BOLD}Best Solution (score: {best.value_avg:.3f}){RESET}")
        print(f"{DIM}Commit: {best.commit_hash}{RESET}")
        print(f"{DIM}Depth: {best.depth}, Visits: {best.visits}{RESET}")
        print()

        # Metrics
        if best.evaluation:
            print(f"{BOLD}Metrics:{RESET}")
            for name, value in best.evaluation.metrics.items():
                print(f"  {name}: {value:.3f}")
            print()

        # Code
        print(f"{BOLD}Code:{RESET}")
        print(Display.code(best.code))

    def _show_alternatives(self):
        """Show alternative solutions."""
        if not self.current_solution:
            print(f"{YELLOW}No solutions found yet.{RESET}")
            return

        alternatives = self.current_solution.get_alternative_solutions(5)

        if not alternatives:
            print(f"{YELLOW}No alternative solutions available.{RESET}")
            return

        print(f"\n{BOLD}Alternative Solutions:{RESET}\n")

        for i, alt in enumerate(alternatives):
            print(f"{BOLD}Alternative {i+1}{RESET} (score: {alt.value_avg:.3f})")

            if alt.feature_coords:
                print(f"{DIM}Features: {alt.feature_coords.to_key()}{RESET}")

            if alt.action:
                print(f"{DIM}Action: {alt.action.description}{RESET}")

            print(Display.code(alt.code[:300] + "..."))
            print()

    def _adopt_best(self):
        """Adopt the best solution."""
        if not self.current_solution or not self.current_solution.best_node:
            print(f"{YELLOW}No solution to adopt.{RESET}")
            return

        try:
            self.controller.git_manager.checkout_main()
            self.controller.git_manager.merge_to_main(
                self.current_solution.best_node.commit_hash,
                f"MCTS: adopted best solution ({self.current_solution.best_score:.3f})"
            )
            print(f"{GREEN}✓ Best solution merged to main branch!{RESET}")
        except Exception as e:
            print(f"{RED}Failed to merge: {e}{RESET}")

    def _continue_exploration(self):
        """Continue exploration."""
        if not self.controller:
            print(f"{YELLOW}No active session.{RESET}")
            print("Start a code task to begin exploration.")
            return

        continue_msg = (
            f"{DIM}Continuing exploration for "
            f"{self.config.max_iterations} more iterations...{RESET}"
        )
        print(continue_msg)

        # Reset iteration counter
        self.controller.iteration = 0
        self.controller.running = True

        # Continue
        solution = self.controller.solve(
            self.controller.search_tree.goal,
            initial_code=self.current_solution.best_code if self.current_solution else ""
        )

        self.current_solution = solution
        print(f"{GREEN}✓ Exploration complete. Best score: {solution.best_score:.3f}{RESET}")

    def _show_report(self):
        """Show full exploration report."""
        if not self.current_solution:
            print(f"{YELLOW}No exploration report available.{RESET}")
            return

        print(self.current_solution.get_report())

    def _clear_conversation(self):
        """Clear conversation history."""
        if self.agent:
            self.agent.clear_history()
            print(f"{GREEN}✓ Conversation history cleared.{RESET}")
        else:
            print(f"{YELLOW}No conversation history to clear.{RESET}")

    def _show_config(self):
        """Show current configuration."""
        print(f"\n{BOLD}Configuration:{RESET}\n")

        config_dict = self.config.to_dict()

        for key, value in config_dict.items():
            if key == "llm":
                print(f"{BOLD}LLM:{RESET}")
                for k, v in value.items():
                    if k == "api_key" and v:
                        v = f"{v[:8]}..."  # Truncate API key
                    print(f"  {k}: {v}")
            else:
                print(f"{BOLD}{key}:{RESET} {value}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ALPHACODE: MCTS-based code exploration"
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config file"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        help="LLM model to use"
    )

    parser.add_argument(
        "--iterations", "-i",
        type=int,
        help="Max iterations"
    )

    parser.add_argument(
        "--working-dir", "-d",
        type=str,
        help="Working directory"
    )

    parser.add_argument(
        "--api-key", "-k",
        type=str,
        help="LLM API key (or set NVIDIA_API_KEY/OPENAI_API_KEY env var)"
    )

    args = parser.parse_args()

    # Determine working directory first
    working_dir = os.path.abspath(args.working_dir) if args.working_dir else os.getcwd()

    # Change directory early
    if args.working_dir:
        os.chdir(working_dir)

    # Load config - try multiple locations
    config = None

    # 1. Explicit config file
    if args.config:
        config = MCTSConfig.from_yaml(args.config)

    # 2. Working directory config (now we're in working dir)
    elif os.path.exists("config.yaml"):
        config = MCTSConfig.from_yaml("config.yaml")

    # 3. Project root config (where the package is installed)
    else:
        try:
            import alphacode
            package_dir = os.path.dirname(alphacode.__file__)
            project_root = os.path.dirname(package_dir)
            config_path = os.path.join(project_root, "config.yaml")
            if os.path.exists(config_path):
                config = MCTSConfig.from_yaml(config_path)
        except Exception:
            pass

    # 4. Default config
    if config is None:
        config = MCTSConfig()

    # Override with command line args
    if args.model:
        config.llm.model = args.model

    if args.api_key:
        config.llm.api_key = args.api_key

    if args.iterations:
        config.max_iterations = args.iterations

    # Ensure API key is set
    if not config.llm.api_key:
        config.llm.api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")

    # Run CLI
    cli = MCTSCli(config)
    cli.run()


if __name__ == "__main__":
    main()
