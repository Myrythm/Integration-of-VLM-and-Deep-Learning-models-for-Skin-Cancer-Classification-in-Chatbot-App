import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@dataclass
class Turn:
    question: str
    answer: str

    def to_messages(self) -> list[BaseMessage]:
        return [HumanMessage(content=self.question), AIMessage(content=self.answer)]


@dataclass
class _Session:
    turns: list[Turn] = field(default_factory=list)
    last_access: float = 0.0


class SessionMemory:
    """Bounded per-session conversation buffer, thread-safe for asyncio.

    Turns are bounded per session (``max_turns``). Sessions themselves are bounded
    globally: idle sessions past ``ttl_seconds`` are dropped, then least-recently
    used sessions are dropped while the store exceeds ``max_sessions``. In
    production, swap for a Redis-backed implementation.
    """

    def __init__(
        self,
        max_turns: int = 6,
        max_sessions: int = 1000,
        ttl_seconds: float = 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._ttl = ttl_seconds
        self._clock = clock
        self._store: "OrderedDict[str, _Session]" = OrderedDict()
        self._lock = asyncio.Lock()

    def _evict(self, now: float) -> None:
        expired = [sid for sid, s in self._store.items() if now - s.last_access > self._ttl]
        for sid in expired:
            del self._store[sid]
        while len(self._store) > self._max_sessions:
            self._store.popitem(last=False)  # drop least-recently-used

    async def add_turn(self, session_id: str, question: str, answer: str) -> None:
        async with self._lock:
            now = self._clock()
            session = self._store.get(session_id)
            if session is None:
                session = _Session()
                self._store[session_id] = session
            session.turns.append(Turn(question, answer))
            if len(session.turns) > self._max_turns:
                session.turns = session.turns[-self._max_turns :]
            session.last_access = now
            self._store.move_to_end(session_id)
            self._evict(now)

    async def get_history(self, session_id: str) -> list[BaseMessage]:
        async with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return []
            session.last_access = self._clock()
            self._store.move_to_end(session_id)
            messages: list[BaseMessage] = []
            for t in session.turns:
                messages.extend(t.to_messages())
            return messages

    def reset(self, session_id: str) -> None:
        self._store.pop(session_id, None)
