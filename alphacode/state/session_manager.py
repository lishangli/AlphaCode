"""
Session Manager for ALPHACODE.

Manages conversation sessions using Git for persistence and traceability.
Each session is a Git branch with commits for each message/tool call.
"""

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from alphacode.state.git_manager import GitStateManager

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Session status."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ARCHIVED = "archived"


class SessionIntent(Enum):
    """Session intent type."""
    CONVERSATION = "conversation"
    CODE_GENERATION = "code_generation"
    MCTS_EXPLORATION = "mcts_exploration"


@dataclass
class SessionEvent:
    """A single event in the session."""
    timestamp: str
    event_type: str  # message, tool_call, mcts_iteration
    role: str = ""  # user, assistant, tool, system
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionEvent":
        return cls(**data)


@dataclass
class SessionState:
    """
    Session state.

    Represents a complete conversation/code generation session.
    """
    session_id: str
    branch_name: str
    intent: str
    status: str = "active"
    created_at: str = ""
    last_active: str = ""
    message_count: int = 0
    tool_calls_count: int = 0
    mcts_iterations: int = 0
    goal: str = ""
    best_score: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.last_active:
            self.last_active = self.created_at

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        return cls(**data)

    def update_activity(self):
        """Update last active timestamp."""
        self.last_active = datetime.now().isoformat()


class SessionManager:
    """
    Git-based Session Manager.

    Features:
    - Create/restore conversation sessions
    - Record messages to Git commits
    - Record tool calls with results
    - Full history traceability
    - Session rollback capability

    Directory structure:
    .alphacode/
    ├── sessions/
    │   ├── {session_id}/
    │   │   ├── metadata.json      # Session metadata
    │   │   ├── conversation.json   # Message history
    │   │   └── tool_calls.json     # Tool call records
    └── cache/
        └── evaluations.db          # Evaluation cache
    """

    def __init__(self, root_path: str = None, git_manager: GitStateManager = None):
        """
        Initialize session manager.

        Args:
            root_path: Root directory path
            git_manager: Optional Git state manager
        """
        self.root_path = root_path or os.getcwd()
        self.git_manager = git_manager or GitStateManager(root_path=self.root_path)

        # Session storage paths
        self.alphacode_dir = os.path.join(self.root_path, ".alphacode")
        self.sessions_dir = os.path.join(self.alphacode_dir, "sessions")
        self.cache_dir = os.path.join(self.alphacode_dir, "cache")

        # Ensure directories exist
        os.makedirs(self.sessions_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Current active session
        self.current_session: SessionState | None = None
        self._events: list[SessionEvent] = []

    def create_session(
        self,
        intent: str = "conversation",
        goal: str = "",
    ) -> SessionState:
        """
        Create a new session.

        Creates:
        - Git branch for the session
        - Session directory with metadata
        - Empty conversation file

        Args:
            intent: Session intent (conversation/code_generation/mcts_exploration)
            goal: Optional goal for the session

        Returns:
            SessionState for the new session
        """
        # Generate session ID
        session_id = str(uuid.uuid4())[:8]
        branch_name = f"session/{session_id}"

        # Create Git branch
        try:
            self.git_manager.create_branch(branch_name)
        except Exception as e:
            logger.warning(f"Could not create branch: {e}")

        # Create session directory
        session_dir = os.path.join(self.sessions_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Create session state
        state = SessionState(
            session_id=session_id,
            branch_name=branch_name,
            intent=intent,
            goal=goal,
        )

        # Save metadata
        self._save_metadata(state)

        # Initialize conversation file
        self._init_conversation_file(session_id)

        # Commit initial state
        self.git_manager.snapshot(f"Session {session_id} started: {intent}")

        # Set as current session
        self.current_session = state
        self._events = []

        logger.info(f"Created session {session_id} (intent={intent})")

        return state

    def record_message(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] = None,
    ) -> SessionEvent:
        """
        Record a message in the session.

        Args:
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata

        Returns:
            SessionEvent created
        """
        if not self.current_session:
            logger.warning("No active session, creating one")
            self.create_session()

        # Create event
        event = SessionEvent(
            timestamp=datetime.now().isoformat(),
            event_type="message",
            role=role,
            content=content,
            metadata=metadata or {},
        )

        # Add to events
        self._events.append(event)

        # Append to conversation file
        self._append_to_conversation(event)

        # Git commit
        commit_msg = f"{role}: {content[:50]}{'...' if len(content) > 50 else ''}"
        self.git_manager.snapshot(commit_msg)

        # Update session state
        self.current_session.message_count += 1
        self.current_session.update_activity()
        self._save_metadata(self.current_session)

        logger.debug(f"Recorded message: {role} - {content[:30]}...")

        return event

    def record_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str,
        success: bool = True,
    ) -> SessionEvent:
        """
        Record a tool call in the session.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool result
            success: Whether the call succeeded

        Returns:
            SessionEvent created
        """
        if not self.current_session:
            logger.warning("No active session")
            return None

        # Create event
        event = SessionEvent(
            timestamp=datetime.now().isoformat(),
            event_type="tool_call",
            role="tool",
            content=f"{tool_name}({args})",
            metadata={
                "tool": tool_name,
                "args": args,
                "result": result[:500] if result else "",
                "success": success,
            },
        )

        # Add to events
        self._events.append(event)

        # Append to tool calls file
        self._append_to_tool_calls(event)

        # Save result to separate file for large outputs
        if result and len(result) > 500:
            result_file = os.path.join(
                self.sessions_dir,
                self.current_session.session_id,
                f"tool_result_{self.current_session.tool_calls_count}.txt"
            )
            with open(result_file, "w") as f:
                f.write(result)
            self.git_manager.snapshot(f"Tool: {tool_name}(result saved)")

        # Update session state
        self.current_session.tool_calls_count += 1
        self.current_session.update_activity()
        self._save_metadata(self.current_session)

        logger.debug(f"Recorded tool call: {tool_name}")

        return event

    def record_mcts_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> SessionEvent:
        """
        Record an MCTS exploration event.

        Args:
            event_type: Type of MCTS event (expand/evaluate/backpropagate)
            data: Event data

        Returns:
            SessionEvent created
        """
        if not self.current_session:
            self.create_session(intent="mcts_exploration")

        # Create event
        event = SessionEvent(
            timestamp=datetime.now().isoformat(),
            event_type=f"mcts_{event_type}",
            role="system",
            content=json.dumps(data),
            metadata=data,
        )

        # Add to events
        self._events.append(event)

        # Update iteration count if this is an iteration event
        if event_type == "iteration":
            self.current_session.mcts_iterations += 1

        # Update best score if provided
        if "score" in data and data["score"] > self.current_session.best_score:
            self.current_session.best_score = data["score"]

        self.current_session.update_activity()
        self._save_metadata(self.current_session)

        return event

    def restore_session(self, session_id: str) -> SessionState:
        """
        Restore a previous session.

        Args:
            session_id: Session ID to restore

        Returns:
            Restored SessionState
        """
        # Load metadata
        state = self._load_metadata(session_id)

        if not state:
            raise ValueError(f"Session {session_id} not found")

        # Checkout Git branch
        try:
            self.git_manager.checkout_branch(state.branch_name)
        except Exception as e:
            logger.warning(f"Could not checkout branch: {e}")

        # Load events
        self._events = self._load_conversation(session_id)

        # Set as current
        self.current_session = state
        state.status = "active"

        logger.info(f"Restored session {session_id}")

        return state

    def get_session_history(self, session_id: str = None) -> list[SessionEvent]:
        """
        Get session conversation history.

        Args:
            session_id: Session ID (default: current session)

        Returns:
            List of SessionEvents
        """
        sid = session_id or (self.current_session.session_id if self.current_session else None)
        if not sid:
            return []

        return self._load_conversation(sid)

    def get_tool_call_history(self, session_id: str = None) -> list[SessionEvent]:
        """
        Get session tool call history.

        Args:
            session_id: Session ID (default: current session)

        Returns:
            List of tool call events
        """
        sid = session_id or (self.current_session.session_id if self.current_session else None)
        if not sid:
            return []

        return self._load_tool_calls(sid)

    def list_sessions(self, status: str = None) -> list[SessionState]:
        """
        List all sessions.

        Args:
            status: Filter by status (optional)

        Returns:
            List of SessionState
        """
        sessions = []

        if not os.path.exists(self.sessions_dir):
            return sessions

        for session_id in os.listdir(self.sessions_dir):
            session_dir = os.path.join(self.sessions_dir, session_id)
            if os.path.isdir(session_dir):
                state = self._load_metadata(session_id)
                if state:
                    if status is None or state.status == status:
                        sessions.append(state)

        # Sort by last active (most recent first)
        sessions.sort(key=lambda s: s.last_active, reverse=True)

        return sessions

    def complete_session(self, session_id: str = None):
        """
        Mark session as completed.

        Args:
            session_id: Session ID (default: current session)
        """
        state = self._get_session_state(session_id)
        if state:
            state.status = "completed"
            self._save_metadata(state)
            logger.info(f"Session {state.session_id} completed")

    def archive_session(self, session_id: str = None):
        """
        Archive session (merge to main branch).

        Args:
            session_id: Session ID (default: current session)
        """
        state = self._get_session_state(session_id)
        if not state:
            return

        # Merge to main
        try:
            self.git_manager.checkout_main()
            self.git_manager.merge_to_main(
                commit_hash=None,  # Current branch tip
                message=f"Archive session {state.session_id}: {state.intent}"
            )
            state.status = "archived"
            self._save_metadata(state)
            logger.info(f"Session {state.session_id} archived")
        except Exception as e:
            logger.error(f"Failed to archive session: {e}")

    def rollback_to_event(self, event_index: int) -> bool:
        """
        Rollback session to a specific event.

        Args:
            event_index: Index of event to rollback to

        Returns:
            True if successful
        """
        if not self.current_session or event_index >= len(self._events):
            return False

        # Truncate events
        self._events = self._events[:event_index + 1]

        # Rewrite conversation file
        self._rewrite_conversation()

        # Git reset (find the commit)
        # This is a simplified implementation
        logger.info(f"Rolled back to event {event_index}")

        return True

    def export_session(self, session_id: str = None) -> dict[str, Any]:
        """
        Export session data for backup/analysis.

        Args:
            session_id: Session ID (default: current session)

        Returns:
            Dict with all session data
        """
        state = self._get_session_state(session_id)
        if not state:
            return {}

        return {
            "metadata": state.to_dict(),
            "events": [e.to_dict() for e in self._events],
            "tool_calls": [e.to_dict() for e in self._events if e.event_type == "tool_call"],
        }

    # ==================== Private methods ====================

    def _get_session_state(self, session_id: str = None) -> SessionState | None:
        """Get session state by ID or current session."""
        if session_id:
            return self._load_metadata(session_id)
        return self.current_session

    def _save_metadata(self, state: SessionState):
        """Save session metadata."""
        metadata_file = os.path.join(
            self.sessions_dir,
            state.session_id,
            "metadata.json"
        )
        with open(metadata_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def _load_metadata(self, session_id: str) -> SessionState | None:
        """Load session metadata."""
        metadata_file = os.path.join(
            self.sessions_dir,
            session_id,
            "metadata.json"
        )
        if not os.path.exists(metadata_file):
            return None

        with open(metadata_file) as f:
            data = json.load(f)
            return SessionState.from_dict(data)

    def _init_conversation_file(self, session_id: str):
        """Initialize empty conversation file."""
        conv_file = os.path.join(
            self.sessions_dir,
            session_id,
            "conversation.json"
        )
        with open(conv_file, "w") as f:
            json.dump([], f)

    def _append_to_conversation(self, event: SessionEvent):
        """Append event to conversation file."""
        conv_file = os.path.join(
            self.sessions_dir,
            self.current_session.session_id,
            "conversation.json"
        )

        # Load existing
        events = []
        if os.path.exists(conv_file):
            with open(conv_file) as f:
                events = json.load(f)

        # Append
        events.append(event.to_dict())

        # Save
        with open(conv_file, "w") as f:
            json.dump(events, f, indent=2)

    def _append_to_tool_calls(self, event: SessionEvent):
        """Append tool call to tool calls file."""
        tc_file = os.path.join(
            self.sessions_dir,
            self.current_session.session_id,
            "tool_calls.json"
        )

        # Load existing
        events = []
        if os.path.exists(tc_file):
            with open(tc_file) as f:
                events = json.load(f)

        # Append
        events.append(event.to_dict())

        # Save
        with open(tc_file, "w") as f:
            json.dump(events, f, indent=2)

    def _load_conversation(self, session_id: str) -> list[SessionEvent]:
        """Load conversation events."""
        conv_file = os.path.join(
            self.sessions_dir,
            session_id,
            "conversation.json"
        )
        if not os.path.exists(conv_file):
            return []

        with open(conv_file) as f:
            events = json.load(f)
            return [SessionEvent.from_dict(e) for e in events]

    def _load_tool_calls(self, session_id: str) -> list[SessionEvent]:
        """Load tool call events."""
        tc_file = os.path.join(
            self.sessions_dir,
            session_id,
            "tool_calls.json"
        )
        if not os.path.exists(tc_file):
            return []

        with open(tc_file) as f:
            events = json.load(f)
            return [SessionEvent.from_dict(e) for e in events]

    def _rewrite_conversation(self):
        """Rewrite conversation file from current events."""
        if not self.current_session:
            return

        conv_file = os.path.join(
            self.sessions_dir,
            self.current_session.session_id,
            "conversation.json"
        )

        with open(conv_file, "w") as f:
            json.dump([e.to_dict() for e in self._events], f, indent=2)
