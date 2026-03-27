"""
Action expander for MCTS-Agent.

Generates candidate actions using LLM.
"""

import json
import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional

from alphacode.config import MCTSConfig
from alphacode.core.node import MCTSNode, Action
from alphacode.llm.client import LLMClient
from alphacode.llm.prompts import PromptBuilder

logger = logging.getLogger(__name__)


class ActionExpander:
    """
    Action expander using LLM.
    
    Generates candidate improvement actions.
    """
    
    def __init__(
        self, 
        config: MCTSConfig,
        llm_client: LLMClient = None,
        prompt_builder: PromptBuilder = None
    ):
        self.config = config
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder()
    
    def generate_actions(
        self,
        prompt: Dict[str, str],
        num_actions: int = None,
    ) -> List[Action]:
        """
        Generate actions from prompt.
        
        Args:
            prompt: Dict with 'system' and 'user' keys
            num_actions: Number of actions to generate
            
        Returns:
            List of Action objects
        """
        num_actions = num_actions or self.config.num_actions_per_expand
        
        if not self.llm_client:
            return self._generate_default_actions(num_actions)
        
        try:
            # Call LLM
            response = asyncio.run(
                self.llm_client.generate_json(
                    prompt=prompt["user"],
                    system=prompt["system"],
                    temperature=0.7,
                )
            )
            
            return self._parse_actions(response, num_actions)
            
        except Exception as e:
            logger.warning(f"LLM action generation failed: {e}")
            return self._generate_default_actions(num_actions)
    
    def _parse_actions(
        self, 
        response: Dict[str, Any],
        num_actions: int
    ) -> List[Action]:
        """Parse LLM response into actions."""
        actions = []
        
        action_list = response.get("actions", [])
        
        for i, action_data in enumerate(action_list[:num_actions]):
            try:
                action = Action(
                    id=str(uuid.uuid4())[:8],
                    description=action_data.get("description", f"Action {i+1}"),
                    reasoning=action_data.get("reasoning", ""),
                    tool_calls=action_data.get("tool_calls", []),
                )
                actions.append(action)
            except Exception as e:
                logger.warning(f"Failed to parse action {i}: {e}")
        
        return actions
    
    def _generate_default_actions(self, num_actions: int) -> List[Action]:
        """Generate default actions when LLM is unavailable."""
        actions = []
        
        for i in range(num_actions):
            action = Action(
                id=str(uuid.uuid4())[:8],
                description=f"Default improvement {i+1}",
                reasoning="No LLM available for action generation",
                tool_calls=[],
            )
            actions.append(action)
        
        return actions
    
    def generate_with_templates(
        self,
        goal: str,
        current_code: str,
        node: MCTSNode,
        inspirations: List[MCTSNode],
        previous_attempts: List[Dict],
        artifacts: Dict[str, Any],
    ) -> List[Action]:
        """
        Generate actions using prompt templates.
        
        Args:
            goal: User goal
            current_code: Current code
            node: Current node
            inspirations: Inspiration nodes
            previous_attempts: Previous attempts
            artifacts: Error artifacts
            
        Returns:
            List of actions
        """
        prompt = self.prompt_builder.build_expand_prompt(
            goal=goal,
            current_code=current_code,
            node=node,
            inspirations=inspirations,
            previous_attempts=previous_attempts,
            artifacts=artifacts,
            num_actions=self.config.num_actions_per_expand,
        )
        
        return self.generate_actions(prompt)


class RuleBasedExpander(ActionExpander):
    """
    Rule-based action expander.
    
    Generates actions based on code analysis rules.
    """
    
    def generate_actions(
        self,
        prompt: Dict[str, str],
        num_actions: int = None,
    ) -> List[Action]:
        """Generate actions based on rules."""
        actions = []
        
        # Extract code from prompt
        code = self._extract_code(prompt.get("user", ""))
        
        if code:
            # Generate rule-based actions
            actions.extend(self._suggest_improvements(code))
        
        # Fill remaining with defaults
        while len(actions) < (num_actions or 3):
            actions.append(self._generate_default_actions(1)[0])
        
        return actions[:num_actions or 3]
    
    def _extract_code(self, text: str) -> str:
        """Extract code from text."""
        import re
        
        # Look for code blocks
        match = re.search(r'```\w*\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1)
        
        return ""
    
    def _suggest_improvements(self, code: str) -> List[Action]:
        """Suggest improvements based on code analysis."""
        actions = []
        
        # Check for missing docstrings
        if '"""' not in code and "'''" not in code:
            actions.append(Action(
                id=str(uuid.uuid4())[:8],
                description="Add docstrings to functions",
                reasoning="Functions lack documentation",
                tool_calls=[
                    {
                        "tool": "edit",
                        "args": {
                            "path": "program.py",
                            "old": "def ",
                            "new": '    """TODO: Add docstring"""\ndef '
                        }
                    }
                ]
            ))
        
        # Check for long functions
        lines = code.split("\n")
        if len(lines) > 50:
            actions.append(Action(
                id=str(uuid.uuid4())[:8],
                description="Refactor long function into smaller pieces",
                reasoning="Function is too long, consider breaking it up",
                tool_calls=[]
            ))
        
        # Check for error handling
        if "try:" not in code and "except" not in code:
            actions.append(Action(
                id=str(uuid.uuid4())[:8],
                description="Add error handling",
                reasoning="No error handling detected",
                tool_calls=[]
            ))
        
        return actions