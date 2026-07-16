# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for :class:`StreamPersistenceManager`, the disconnect/cancel layer.

The tests drive scripted inner agents through the manager (wrapped by a real
:class:`AGUIStorageAgent` built by the factory) against the dependency-free
in-memory repositories, and assert the full behaviour matrix:

* normal run -> ``complete`` and every event streamed in order;
* client disconnect (never reading ``events()``) -> still drains to ``complete``;
* explicit cancel -> ``interrupted`` persisted, stream terminated;
* producer exception -> synthesised ``RunErrorEvent`` and the consumer never
  hangs; a terminal ``RunErrorEvent`` from the inner agent persists ``errored``;
* ``cancel`` of an unknown key -> ``False``;
* the registry isolates concurrent runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from collections.abc import Callable
from uuid import UUID
from uuid import uuid4

from ag_ui.core import BaseEvent
from ag_ui.core import RunAgentInput
from ag_ui.core import RunErrorEvent
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent

from datarobot_genai.application_utils.chat_history.ag_ui_storage import AGUIAgent
from datarobot_genai.application_utils.chat_history.ag_ui_storage import AGUIStorageAgent
from datarobot_genai.application_utils.chat_history.ag_ui_storage import ErrorCodes
from datarobot_genai.application_utils.chat_history.models import Message
from datarobot_genai.application_utils.chat_history.models import MessageStatus
from datarobot_genai.application_utils.chat_history.stream_manager import NoMoreEvents
from datarobot_genai.application_utils.chat_history.stream_manager import RunHandle
from datarobot_genai.application_utils.chat_history.stream_manager import StreamPersistenceManager
from tests.application_utils.chat_history.unit.fake_repos import InMemoryChatRepository
from tests.application_utils.chat_history.unit.fake_repos import InMemoryMessageRepository

THREAD = "thread-1"
RUN = "run-1"


# ── Scripted inner agents ──────────────────────────────────────────────────────


class ScriptedAgent(AGUIAgent):
    """An inner agent that emits a fixed list of events."""

    def __init__(self, name: str, events: list[BaseEvent]) -> None:
        super().__init__(name)
        self.events = events

    async def run(self, input: RunAgentInput) -> AsyncIterator[BaseEvent]:
        """Emit the scripted events in order."""
        for event in self.events:
            yield event


class GatedAgent(AGUIAgent):
    """An inner agent that emits pre-events then blocks forever on a gate."""

    def __init__(self, name: str, pre_events: list[BaseEvent], gate: asyncio.Event) -> None:
        super().__init__(name)
        self.pre_events = pre_events
        self.gate = gate

    async def run(self, input: RunAgentInput) -> AsyncIterator[BaseEvent]:
        """Emit the pre-events, then await the gate (released or cancelled)."""
        for event in self.pre_events:
            yield event
        await self.gate.wait()
        yield _run_finished()


class RaisingAgent(AGUIAgent):
    """An inner agent that emits pre-events then raises a raw exception."""

    def __init__(self, name: str, pre_events: list[BaseEvent], exc: Exception) -> None:
        super().__init__(name)
        self.pre_events = pre_events
        self.exc = exc

    async def run(self, input: RunAgentInput) -> AsyncIterator[BaseEvent]:
        """Emit the pre-events, then raise mid-stream (no terminal event)."""
        for event in self.pre_events:
            yield event
        raise self.exc


# ── Builders / helpers ──────────────────────────────────────────────────────────


def _run_started() -> RunStartedEvent:
    return RunStartedEvent(thread_id=THREAD, run_id=RUN)


def _run_finished() -> RunFinishedEvent:
    return RunFinishedEvent(thread_id=THREAD, run_id=RUN)


def _input(thread_id: str = THREAD, run_id: str = RUN) -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
        run_id=run_id,
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props={},
    )


def _text_turn(
    message_id: str = "m1", deltas: tuple[str, ...] = ("Hello ", "world")
) -> list[BaseEvent]:
    events: list[BaseEvent] = [
        _run_started(),
        TextMessageStartEvent(message_id=message_id, role="assistant"),
    ]
    events.extend(TextMessageContentEvent(message_id=message_id, delta=d) for d in deltas)
    events.append(TextMessageEndEvent(message_id=message_id))
    events.append(_run_finished())
    return events


def _factory(
    inner: AGUIAgent,
    chat_repo: InMemoryChatRepository,
    message_repo: InMemoryMessageRepository,
    user_id: UUID,
) -> Callable[[], AGUIStorageAgent]:
    """Return a zero-arg factory building a storage agent around *inner*."""

    def build() -> AGUIStorageAgent:
        return AGUIStorageAgent("assistant", user_id, chat_repo, message_repo, inner)

    return build


async def _wait_until(pred: Callable[[], bool], *, timeout: float = 2.0) -> None:
    elapsed = 0.0
    while not pred():
        if elapsed >= timeout:
            raise AssertionError("condition was not met in time")
        await asyncio.sleep(0.01)
        elapsed += 0.01


def _messages(repo: InMemoryMessageRepository) -> list[Message]:
    return list(repo.messages.values())


async def _collect(handle: RunHandle) -> list[BaseEvent]:
    return [event async for event in handle.events()]


# ── Normal completion ─────────────────────────────────────────────────────────


async def test_normal_run_streams_all_events_and_persists_complete() -> None:
    """GIVEN a normal run WHEN fully consumed THEN every event streams and it persists complete."""
    events = _text_turn()
    message_repo = InMemoryMessageRepository()
    manager = StreamPersistenceManager(
        _factory(ScriptedAgent("inner", events), InMemoryChatRepository(), message_repo, uuid4())
    )

    handle = await manager.run(_input())
    emitted = [event async for event in handle.events()]
    await handle.wait()

    assert [type(e) for e in emitted] == [type(e) for e in events]
    message = _messages(message_repo)[0]
    assert message.content == "Hello world"
    assert message.in_progress is False
    assert message.status == MessageStatus.COMPLETE.value


async def test_wait_returns_after_producer_finishes_and_clears_registry() -> None:
    """GIVEN a finished run WHEN awaited THEN the task is done and the run is unregistered."""
    message_repo = InMemoryMessageRepository()
    manager = StreamPersistenceManager(
        _factory(
            ScriptedAgent("inner", _text_turn()), InMemoryChatRepository(), message_repo, uuid4()
        )
    )

    handle = await manager.run(_input())
    await handle.wait()

    # The producer unregistered itself, so a late cancel finds nothing.
    assert manager.cancel(THREAD, RUN) is False
    assert handle.cancel() is False


# ── Disconnect survival ─────────────────────────────────────────────────────────


async def test_disconnect_survival_drains_to_complete_without_reading_events() -> None:
    """GIVEN a client that never reads events WHEN it waits THEN persistence still completes."""
    message_repo = InMemoryMessageRepository()
    manager = StreamPersistenceManager(
        _factory(
            ScriptedAgent("inner", _text_turn()), InMemoryChatRepository(), message_repo, uuid4()
        )
    )

    handle = await manager.run(_input())
    # Simulate a disconnect: never iterate handle.events(), just await the drain.
    await handle.wait()

    message = _messages(message_repo)[0]
    assert message.content == "Hello world"
    assert message.status == MessageStatus.COMPLETE.value


# ── Cancellation ────────────────────────────────────────────────────────────────


async def test_cancel_persists_interrupted_and_terminates_stream() -> None:
    """GIVEN a mid-stream cancel WHEN finalised THEN records are interrupted and events end."""
    gate = asyncio.Event()
    pre: list[BaseEvent] = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="partial"),
    ]
    message_repo = InMemoryMessageRepository()
    manager = StreamPersistenceManager(
        _factory(GatedAgent("inner", pre, gate), InMemoryChatRepository(), message_repo, uuid4())
    )

    handle = await manager.run(_input())
    collected: list[BaseEvent] = []

    async def consume() -> None:
        async for event in handle.events():
            collected.append(event)

    reader = asyncio.create_task(consume())
    await _wait_until(lambda: len(collected) >= len(pre))

    assert handle.cancel() is True
    await handle.wait()
    await asyncio.wait_for(reader, timeout=2.0)  # events() terminated (never hangs)

    message = _messages(message_repo)[0]
    assert message.content == "partial"
    assert message.in_progress is False
    assert message.status == MessageStatus.INTERRUPTED.value


async def test_cancel_via_manager_key_matches_run_handle() -> None:
    """GIVEN a live run WHEN cancelled by manager key THEN it interrupts, same as the handle."""
    gate = asyncio.Event()
    pre: list[BaseEvent] = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="partial"),
    ]
    message_repo = InMemoryMessageRepository()
    manager = StreamPersistenceManager(
        _factory(GatedAgent("inner", pre, gate), InMemoryChatRepository(), message_repo, uuid4())
    )

    handle = await manager.run(_input())
    collected: list[BaseEvent] = []

    async def consume() -> None:
        async for event in handle.events():
            collected.append(event)

    reader = asyncio.create_task(consume())
    await _wait_until(lambda: len(collected) >= len(pre))

    assert manager.cancel(THREAD, RUN) is True
    await handle.wait()
    await asyncio.wait_for(reader, timeout=2.0)

    assert _messages(message_repo)[0].status == MessageStatus.INTERRUPTED.value


# ── Never-hang: producer exception ───────────────────────────────────────────────


async def test_producer_exception_synthesizes_run_error_and_never_hangs() -> None:
    """GIVEN a producer that raises WHEN consumed THEN a RunError is synthesised and events end."""
    pre: list[BaseEvent] = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="partial"),
    ]
    inner = RaisingAgent("inner", pre, RuntimeError("kaboom"))
    manager = StreamPersistenceManager(
        _factory(inner, InMemoryChatRepository(), InMemoryMessageRepository(), uuid4())
    )

    handle = await manager.run(_input())
    # A hang would never terminate; the timeout is the assertion that it does.
    emitted = await asyncio.wait_for(_collect(handle), timeout=5.0)
    await handle.wait()

    assert isinstance(emitted[-1], RunErrorEvent)
    assert emitted[-1].code == ErrorCodes.INTERNAL_ERROR.value
    assert "kaboom" in emitted[-1].message
    # The pre-events were forwarded before the synthesised terminal error.
    assert [type(e) for e in emitted[:-1]] == [type(e) for e in pre]


async def test_terminal_run_error_persists_errored_without_duplicate_synthesis() -> None:
    """GIVEN an inner RunError WHEN consumed THEN it persists errored and is not duplicated."""
    events: list[BaseEvent] = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="partial"),
        RunErrorEvent(message="boom", code="INTERNAL_ERROR"),
    ]
    message_repo = InMemoryMessageRepository()
    manager = StreamPersistenceManager(
        _factory(ScriptedAgent("inner", events), InMemoryChatRepository(), message_repo, uuid4())
    )

    handle = await manager.run(_input())
    emitted = await asyncio.wait_for(_collect(handle), timeout=5.0)
    await handle.wait()

    # The inner terminal RunError is forwarded exactly once (no extra synthesis).
    assert sum(isinstance(e, RunErrorEvent) for e in emitted) == 1
    message = _messages(message_repo)[0]
    assert message.content == "partial"
    assert message.in_progress is False
    assert message.status == MessageStatus.ERRORED.value
    assert "boom" in (message.error or "")


# ── Unknown-key cancel ────────────────────────────────────────────────────────────


async def test_cancel_unknown_key_returns_false() -> None:
    """GIVEN no matching run WHEN cancel is called THEN it returns False."""
    manager = StreamPersistenceManager(
        _factory(
            ScriptedAgent("inner", []),
            InMemoryChatRepository(),
            InMemoryMessageRepository(),
            uuid4(),
        )
    )

    assert manager.cancel("no-such-thread", "no-such-run") is False


# ── Registry isolation ────────────────────────────────────────────────────────────


async def test_registry_isolates_concurrent_runs() -> None:
    """GIVEN two concurrent runs WHEN one is cancelled THEN the other completes independently."""
    gate_a = asyncio.Event()
    gate_b = asyncio.Event()
    pre_a: list[BaseEvent] = [
        _run_started(),
        TextMessageStartEvent(message_id="a1", role="assistant"),
        TextMessageContentEvent(message_id="a1", delta="alpha"),
    ]
    pre_b: list[BaseEvent] = [
        _run_started(),
        TextMessageStartEvent(message_id="b1", role="assistant"),
        TextMessageContentEvent(message_id="b1", delta="beta"),
    ]
    repo_a = InMemoryMessageRepository()
    repo_b = InMemoryMessageRepository()

    def build_a() -> AGUIStorageAgent:
        return AGUIStorageAgent(
            "assistant", uuid4(), InMemoryChatRepository(), repo_a, GatedAgent("a", pre_a, gate_a)
        )

    def build_b() -> AGUIStorageAgent:
        return AGUIStorageAgent(
            "assistant", uuid4(), InMemoryChatRepository(), repo_b, GatedAgent("b", pre_b, gate_b)
        )

    # A single manager whose factory picks the builder passed as a run arg.
    manager = StreamPersistenceManager(lambda build: build())

    handle_a = await manager.run(_input(thread_id="thread-a", run_id="run-a"), build_a)
    handle_b = await manager.run(_input(thread_id="thread-b", run_id="run-b"), build_b)

    collected_a: list[BaseEvent] = []
    collected_b: list[BaseEvent] = []

    async def read(handle: RunHandle, sink: list[BaseEvent]) -> None:
        async for event in handle.events():
            sink.append(event)

    reader_a = asyncio.create_task(read(handle_a, collected_a))
    reader_b = asyncio.create_task(read(handle_b, collected_b))
    await _wait_until(lambda: len(collected_a) >= len(pre_a) and len(collected_b) >= len(pre_b))

    # Cancel only A; B must be untouched.
    assert manager.cancel("thread-a", "run-a") is True
    assert manager.cancel("thread-c", "run-c") is False  # unknown key
    await handle_a.wait()
    await asyncio.wait_for(reader_a, timeout=2.0)

    # B is still live and registered; release its gate to let it finish.
    assert manager.cancel("thread-b", "run-b") is True
    gate_b.set()
    await handle_b.wait()
    await asyncio.wait_for(reader_b, timeout=2.0)

    assert _messages(repo_a)[0].status == MessageStatus.INTERRUPTED.value
    assert _messages(repo_b)[0].content == "beta"


async def test_finished_run_does_not_unregister_a_key_reused_by_a_later_run() -> None:
    """GIVEN two runs sharing one key WHEN the earlier finishes THEN the later stays registered.

    The earlier run's producer must drop only *its own* registry entry; a later
    run that reused the same key must remain registered and cancellable.
    """
    gate_a = asyncio.Event()
    gate_b = asyncio.Event()
    pre_a: list[BaseEvent] = [
        _run_started(),
        TextMessageStartEvent(message_id="a1", role="assistant"),
    ]
    pre_b: list[BaseEvent] = [
        _run_started(),
        TextMessageStartEvent(message_id="b1", role="assistant"),
    ]

    def build_a() -> AGUIStorageAgent:
        return AGUIStorageAgent(
            "assistant",
            uuid4(),
            InMemoryChatRepository(),
            InMemoryMessageRepository(),
            GatedAgent("a", pre_a, gate_a),
        )

    def build_b() -> AGUIStorageAgent:
        return AGUIStorageAgent(
            "assistant",
            uuid4(),
            InMemoryChatRepository(),
            InMemoryMessageRepository(),
            GatedAgent("b", pre_b, gate_b),
        )

    manager = StreamPersistenceManager(lambda build: build())

    # Run A registers under the shared key and blocks on its gate.
    handle_a = await manager.run(_input(), build_a)
    # Run B reuses A's exact (thread_id, run_id) key, overwriting the registry entry.
    handle_b = await manager.run(_input(), build_b)

    # Let A finish: its producer's finally must not evict B, which now owns the key.
    gate_a.set()
    await handle_a.wait()

    # B is still registered under the shared key and remains cancellable.
    assert manager.cancel(THREAD, RUN) is True
    await handle_b.wait()
    # B unregistered itself on completion, so the key is finally clear.
    assert manager.cancel(THREAD, RUN) is False


# ── Sentinel contract ─────────────────────────────────────────────────────────────


async def test_events_stops_exactly_at_the_sentinel() -> None:
    """GIVEN a completed run WHEN events() is drained THEN it stops at NoMoreEvents."""
    manager = StreamPersistenceManager(
        _factory(
            ScriptedAgent("inner", _text_turn()),
            InMemoryChatRepository(),
            InMemoryMessageRepository(),
            uuid4(),
        )
    )
    handle = await manager.run(_input())

    emitted = [event async for event in handle.events()]
    await handle.wait()

    # No sentinel leaks into the client stream, and the last item is terminal.
    assert not any(isinstance(e, NoMoreEvents) for e in emitted)
    assert isinstance(emitted[-1], RunFinishedEvent)
