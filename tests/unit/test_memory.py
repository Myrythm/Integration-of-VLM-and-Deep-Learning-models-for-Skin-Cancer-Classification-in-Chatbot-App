from services.rag.memory import SessionMemory


async def test_turns_bounded_at_max_turns() -> None:
    mem = SessionMemory(max_turns=6)
    for i in range(10):
        await mem.add_turn("s1", f"q{i}", f"a{i}")
    history = await mem.get_history("s1")
    assert len(history) == 12
    assert history[0].content == "q4"
    assert history[-1].content == "a9"


async def test_get_history_unknown_session_returns_empty() -> None:
    mem = SessionMemory()
    assert await mem.get_history("does-not-exist") == []


async def test_session_count_bounded_lru_eviction() -> None:
    mem = SessionMemory(max_sessions=2)
    await mem.add_turn("s1", "q", "a")
    await mem.add_turn("s2", "q", "a")
    await mem.add_turn("s3", "q", "a")  # oldest (s1) evicted
    assert await mem.get_history("s1") == []
    assert len(await mem.get_history("s2")) == 2
    assert len(await mem.get_history("s3")) == 2


async def test_ttl_eviction_with_injected_clock() -> None:
    clock = {"now": 1000.0}
    mem = SessionMemory(ttl_seconds=100, clock=lambda: clock["now"])
    await mem.add_turn("s1", "q", "a")

    clock["now"] = 1050.0  # 50s idle < ttl
    await mem.add_turn("s2", "q", "a")
    assert len(await mem.get_history("s1")) == 2

    clock["now"] = 1300.0  # s1 now idle past ttl
    await mem.add_turn("s3", "q", "a")  # triggers eviction sweep
    assert await mem.get_history("s1") == []
