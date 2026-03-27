"""
Example: Basic usage of MCTS-Agent.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts_agent import MCTSConfig, MCTSController


def example_basic():
    """Basic usage example."""
    
    # Create configuration
    config = MCTSConfig(
        max_iterations=50,
        num_islands=2,
        target_score=0.8,
    )
    
    # Set API key (or use environment variable)
    # config.llm.api_key = "your-api-key"
    
    # Create controller
    controller = MCTSController(config)
    
    # Solve a problem
    goal = "Implement a function that finds the longest palindrome in a string"
    
    solution = controller.solve(goal)
    
    print(f"\nSolution found with score: {solution.best_score:.3f}")
    print(f"Iterations: {solution.iterations}")
    print(f"Nodes explored: {solution.total_nodes}")
    print(f"\nBest code:\n{solution.best_code[:500]}...")
    
    # Get alternatives
    alternatives = solution.get_alternative_solutions(3)
    print(f"\nFound {len(alternatives)} alternative solutions")
    
    # Generate report
    report = solution.get_report()
    print(f"\n{report}")


def example_with_initial_code():
    """Example with initial code."""
    
    config = MCTSConfig(
        max_iterations=30,
    )
    
    controller = MCTSController(config)
    
    # Initial code to improve
    initial_code = '''
def sort_numbers(numbers):
    """Sort a list of numbers."""
    # TODO: implement properly
    return numbers
'''
    
    goal = "Implement an efficient sorting algorithm"
    
    solution = controller.solve(goal, initial_code=initial_code)
    
    print(f"Improved code:\n{solution.best_code}")


def example_no_llm():
    """Example without LLM (uses mock/simple actions)."""
    
    config = MCTSConfig(
        max_iterations=10,
    )
    
    # Don't set LLM client
    controller = MCTSController(config)
    # controller.llm_client = None  # Will use simple actions
    
    goal = "Add comments to the code"
    
    solution = controller.solve(goal, initial_code="def hello():\n    print('hello')")
    
    print(f"Result: {solution.best_code}")


if __name__ == "__main__":
    print("MCTS-Agent Examples\n")
    print("=" * 50)
    
    # Run without LLM for testing
    print("\nExample without LLM:")
    example_no_llm()