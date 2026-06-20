import asyncio
from collections import defaultdict
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@dataclass
class Turn:
    question: str
    answer: str

    def to_messages(self) -> list[BaseMessage]:
        return [HumanMessage(content=self.question), AIMessage(content=self.answer)]


class SessionMemory:
    """Bounded per-session conversation buffer. Thread-safe for asyncio.
    In production, swap for Redis-backed implementation.
    """

    def __init__(self, max_turns: int = 6) -> None:
        self._max_turns = max_turns
        self._store: dict[str, list[Turn]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def add_turn(self, session_id: str, question: str, answer: str) -> None:
        async with self._lock:
            turns = self._store[session_id]
            turns.append(Turn(question, answer))
            if len(turns) > self._max_turns:
                self._store[session_id] = turns[-self._max_turns :]

    def get_history(self, session_id: str) -> list[BaseMessage]:
        turns = self._store.get(session_id, [])
        messages: list[BaseMessage] = []
        for t in turns:
            messages.extend(t.to_messages())
        return messages

    def reset(self, session_id: str) -> None:
        self._store.pop(session_id, None)
