# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Short-term chat memory for the agent workflow.

The minimum project originally processed each ticket independently. This module
adds an optional session memory that keeps a bounded history of recent user and
assistant messages. The memory is intentionally simple and in-process only:

* it is not written to disk;
* it is lost when the process exits;
* it uses a fixed-size deque to avoid unbounded growth;
* it can be passed into ``run_workflow`` or managed through ``SupportSession``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List


@dataclass(frozen=True)
class ChatMessage:
    """A single chat-memory message."""

    role: str
    content: str


class ShortTermChatMemory:
    """Bounded in-memory chat history.

    Args:
        max_messages: Maximum number of messages to keep. Because both user and
            assistant messages are stored, ``max_messages=10`` means roughly the
            last five request/response turns.
    """

    def __init__(self, max_messages: int = 10) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        self.max_messages = max_messages
        self._messages: Deque[ChatMessage] = deque(maxlen=max_messages)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to memory."""
        normalized_role = role.strip().lower() or "system"
        normalized_content = content.strip()
        if not normalized_content:
            return
        self._messages.append(ChatMessage(role=normalized_role, content=normalized_content))

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        self.add_message("assistant", content)

    def clear(self) -> None:
        """Clear all stored messages."""
        self._messages.clear()

    def to_messages(self) -> List[ChatMessage]:
        """Return a list copy of stored messages."""
        return list(self._messages)

    def to_dicts(self) -> List[dict]:
        """Return memory as a list of serializable dictionaries."""
        return [{"role": message.role, "content": message.content} for message in self._messages]

    def to_text(self) -> str:
        """Render memory as a compact prompt-ready text block."""
        if not self._messages:
            return "No previous conversation."
        lines = []
        for message in self._messages:
            role = "User" if message.role == "user" else "Assistant"
            lines.append(f"{role}: {message.content}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self) -> Iterable[ChatMessage]:
        return iter(self._messages)
