"""Sliding-window conversation memory for multi-turn chat."""

from typing import List, Dict


class ConversationMemory:
    """Maintains a sliding window of recent conversation turns."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_turn(self, role: str, content: str) -> None:
        """Add a message to history."""
        self.history.append({"role": role, "content": content})
        # Keep only the last N turns (each turn = 1 message)
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_history(self) -> List[Dict[str, str]]:
        """Return the current conversation history."""
        return list(self.history)

    def clear(self) -> None:
        """Reset conversation history."""
        self.history = []
