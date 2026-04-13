"""
Smoke test for Brief 07: EnvironmentWatcher
Run: python test_brief07.py
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import asyncio
import os
import tempfile
import time

# Minimal stubs so we don't need the full model stack
class FakeDB:
    def __init__(self, n=0):
        self.memory = {"episodic_log": [{"id": i} for i in range(n)]}

class FakeAgent:
    def __init__(self, n=0):
        self.db = FakeDB(n)


# ---------------------------------------------------------------------------
# Bootstrap path so relative imports work
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from core_logic.task_graph import TaskGraph
from core_logic.event_queue import EventQueue
from core_logic.environment import EnvironmentWatcher, IGNORED_PATTERNS


PASS = 0
FAIL = 0

def check(label: str, cond: bool):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


# ---------------------------------------------------------------------------
# Test 1: Lifecycle — start / stop without crashing
# ---------------------------------------------------------------------------
async def test_lifecycle():
    print("\n--- Test 1: Lifecycle ---")
    tg = TaskGraph(db_path=":memory:")
    eq = EventQueue()
    agent = FakeAgent(n=0)
    loop = asyncio.get_event_loop()

    watcher = EnvironmentWatcher(
        task_graph=tg,
        event_queue=eq,
        agent=agent,
        event_loop=loop,
        watch_paths=["."],
    )

    await watcher.start()
    check("watcher._running is True after start()", watcher._running is True)
    check("Observer started", watcher._observer is not None and watcher._observer.is_alive())

    await watcher.stop()
    check("watcher._running is False after stop()", watcher._running is False)
    check("Observer stopped", watcher._observer is None)
    tg.close()


# ---------------------------------------------------------------------------
# Test 2: Memory growth threshold
# ---------------------------------------------------------------------------
async def test_memory_growth():
    print("\n--- Test 2: Memory growth threshold ---")
    tg = TaskGraph(db_path=":memory:")
    eq = EventQueue()
    agent = FakeAgent(n=3)  # start with 3 episodes
    loop = asyncio.get_event_loop()

    watcher = EnvironmentWatcher(
        task_graph=tg,
        event_queue=eq,
        agent=agent,
        event_loop=loop,
        watch_paths=["."],
        memory_growth_threshold=5,
    )
    await watcher.start()

    # Add 4 more episodes — still below threshold (3 + 4 = 7, growth = 4 < 5)
    agent.db.memory["episodic_log"] = [{"id": i} for i in range(7)]
    await watcher.check_memory_growth()
    check("No trigger below threshold (growth 4 < 5)", eq.is_empty())

    # Add 1 more — crosses threshold (3 + 5 = 8, growth = 5 >= 5)
    agent.db.memory["episodic_log"] = [{"id": i} for i in range(8)]
    await watcher.check_memory_growth()
    events = await eq.drain()
    check("memory_growth trigger fires at threshold", len(events) == 1)
    check("event type is system_trigger", events[0].type == "system_trigger")
    check("trigger payload has memory_growth key", events[0].payload.get("trigger") == "memory_growth")
    check("task written to TaskGraph (write-ahead)", tg.get_task(events[0].payload["task_id"]) is not None)

    # Snapshot should have advanced — no double-fire
    await watcher.check_memory_growth()
    check("No double-fire after snapshot advance", eq.is_empty())

    await watcher.stop()
    tg.close()


# ---------------------------------------------------------------------------
# Test 3: Interaction density counter
# ---------------------------------------------------------------------------
async def test_interaction_density():
    print("\n--- Test 3: Interaction density counter ---")
    tg = TaskGraph(db_path=":memory:")
    eq = EventQueue()
    agent = FakeAgent()
    loop = asyncio.get_event_loop()

    watcher = EnvironmentWatcher(
        task_graph=tg,
        event_queue=eq,
        agent=agent,
        event_loop=loop,
        watch_paths=["."],
        interaction_density_threshold=3,
    )
    await watcher.start()

    # 2 interactions — below threshold
    watcher.notify_interaction()
    watcher.notify_interaction()
    await asyncio.sleep(0.05)  # let run_coroutine_threadsafe complete
    check("No trigger before threshold", eq.is_empty())

    # 3rd interaction — threshold reached
    watcher.notify_interaction()
    await asyncio.sleep(0.05)
    events = await eq.drain()
    check("interaction_density trigger fires at threshold", len(events) == 1)
    check("trigger is interaction_density", events[0].payload.get("trigger") == "interaction_density")
    check("counter reset to 0 after trigger", watcher._interaction_count == 0)

    await watcher.stop()
    tg.close()


# ---------------------------------------------------------------------------
# Test 4: Write-ahead contract — Task in SQLite before event emitted
# ---------------------------------------------------------------------------
async def test_write_ahead():
    print("\n--- Test 4: Write-ahead contract ---")
    import sqlite3

    db_path = os.path.join(tempfile.gettempdir(), "test_brief07_wa.db")
    try:
        tg = TaskGraph(db_path=db_path)
        eq = EventQueue()
        agent = FakeAgent()
        loop = asyncio.get_event_loop()

        watcher = EnvironmentWatcher(
            task_graph=tg,
            event_queue=eq,
            agent=agent,
            event_loop=loop,
        )
        await watcher.start()

        # Manually trigger file_change
        await watcher._on_file_changed("core_logic/some_module.py", "modified")
        events = await eq.drain()

        check("file_change event emitted", len(events) >= 1)
        if events:
            task_id = events[0].payload.get("task_id")
            check("task_id in event payload", task_id is not None)

            # Check SQLite directly
            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT id FROM tasks WHERE id=?", (task_id,)).fetchone()
            conn.close()
            check("Task persisted to SQLite before event delivered", row is not None)

        await watcher.stop()
        tg.close()
    finally:
        try:
            os.remove(db_path)
            os.remove(db_path + "-wal")
            os.remove(db_path + "-shm")
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Test 5: Noise filtering — ignored patterns produce no triggers
# ---------------------------------------------------------------------------
async def test_noise_filtering():
    print("\n--- Test 5: Noise filtering ---")
    tg = TaskGraph(db_path=":memory:")
    eq = EventQueue()
    agent = FakeAgent()
    loop = asyncio.get_event_loop()

    watcher = EnvironmentWatcher(
        task_graph=tg,
        event_queue=eq,
        agent=agent,
        event_loop=loop,
    )
    await watcher.start()

    noisy_paths = [
        "core_logic/__pycache__/agent.cpython-312.pyc",
        "core_logic/tasks.db",
        "core_logic/tasks.db-wal",
        "core_logic/tasks.db-shm",
        "logs/session_20260411.log",
    ]

    for path in noisy_paths:
        await watcher._on_file_changed(path, "modified")

    check("No triggers emitted for noisy paths", eq.is_empty())

    # A real code file should trigger
    await watcher._on_file_changed("core_logic/agent.py", "modified")
    events = await eq.drain()
    check("Real .py file produces file_change trigger", len(events) == 1)

    await watcher.stop()
    tg.close()


# ---------------------------------------------------------------------------
# Test 6: memory.json change routes to check_memory_growth, not file_change
# ---------------------------------------------------------------------------
async def test_memory_json_routing():
    print("\n--- Test 6: memory.json routes to growth check ---")
    tg = TaskGraph(db_path=":memory:")
    eq = EventQueue()
    agent = FakeAgent(n=0)
    loop = asyncio.get_event_loop()

    watcher = EnvironmentWatcher(
        task_graph=tg,
        event_queue=eq,
        agent=agent,
        event_loop=loop,
        memory_growth_threshold=2,
    )
    await watcher.start()

    # memory.json change with growth below threshold → no trigger
    agent.db.memory["episodic_log"] = [{"id": 0}]  # growth = 1 < 2
    await watcher._on_file_changed("core_logic/memory.json", "modified")
    check("memory.json change below threshold -> no trigger", eq.is_empty())

    # memory.json change with growth at threshold → memory_growth trigger, NOT file_change
    agent.db.memory["episodic_log"] = [{"id": 0}, {"id": 1}]  # growth = 2 >= 2
    await watcher._on_file_changed("core_logic/memory.json", "modified")
    events = await eq.drain()
    check("memory.json at threshold -> memory_growth trigger", len(events) == 1)
    check("trigger is memory_growth (not file_change)", events[0].payload.get("trigger") == "memory_growth")

    await watcher.stop()
    tg.close()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
async def main():
    print("=" * 60)
    print("Brief 07 — EnvironmentWatcher Smoke Tests")
    print("=" * 60)
    await test_lifecycle()
    await test_memory_growth()
    await test_interaction_density()
    await test_write_ahead()
    await test_noise_filtering()
    await test_memory_json_routing()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    return FAIL


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
