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

"""Acceptance tests for the AG-UI chat-history layer against the live Memory Service.

These tests exercise the full chat-history stack — :class:`ChatRepository` /
:class:`MessageRepository`, the :class:`AGUIStorageAgent` state machine and the
:class:`StreamPersistenceManager` — end to end against a **live** DataRobot
Memory Service.  Because we do not wire a real inner agent, each scenario drives
a *scripted* :class:`AGUIAgent` that emits a canned AG-UI event stream.

They are **skipped by default** (:func:`pytest.mark.integration`) and require
credentials:

.. code-block:: bash

    export DATAROBOT_ENDPOINT="https://app.datarobot.com/api/v2"
    export DATAROBOT_API_TOKEN="<your-token>"

    uv run pytest tests/application_utils/chat_history/acceptance -m integration -vv

Each test creates a unique :class:`DRMemorySpace` and deletes it on teardown, so
the suite leaves no residue and can be re-run.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from collections.abc import Callable

import pytest
from ag_ui.core import BaseEvent
from ag_ui.core import RunAgentInput
from ag_ui.core import RunErrorEvent
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ThinkingEndEvent
from ag_ui.core import ThinkingStartEvent
from ag_ui.core import ThinkingTextMessageContentEvent
from ag_ui.core import ThinkingTextMessageEndEvent
from ag_ui.core import ThinkingTextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallResultEvent
from ag_ui.core import ToolCallStartEvent
from ag_ui.core import UserMessage

from datarobot_genai.application_utils.chat_history import AGUIAgent
from datarobot_genai.application_utils.chat_history import AGUIStorageAgent
from datarobot_genai.application_utils.chat_history import ChatCreate
from datarobot_genai.application_utils.chat_history import ChatRepository
from datarobot_genai.application_utils.chat_history import ChatSessionRegistry
from datarobot_genai.application_utils.chat_history import ErrorCodes
from datarobot_genai.application_utils.chat_history import MessageRepository
from datarobot_genai.application_utils.chat_history import MessageStatus
from datarobot_genai.application_utils.chat_history import Reasoning
from datarobot_genai.application_utils.chat_history import Role
from datarobot_genai.application_utils.chat_history import RunHandle
from datarobot_genai.application_utils.chat_history import StreamPersistenceManager
from datarobot_genai.application_utils.chat_history import ToolCall
from datarobot_genai.application_utils.persistence import DRMemoryServiceClient
from datarobot_genai.application_utils.persistence import DRMemorySpace

pytestmark = pytest.mark.integration

_HAS_CREDS = bool(os.getenv("DATAROBOT_API_TOKEN")) and bool(os.getenv("DATAROBOT_ENDPOINT"))

skip_unless_live = pytest.mark.skipif(
    not _HAS_CREDS,
    reason="Requires DATAROBOT_ENDPOINT and DATAROBOT_API_TOKEN to be set.",
)


# ── Scripted inner agents ──────────────────────────────────────────────────────


class ScriptedAgent(AGUIAgent):
    """An inner agent that emits a fixed event list and records its (injected) input."""

    def __init__(self, name: str, events: list[BaseEvent]) -> None:
        super().__init__(name)
        self.events = events
        self.seen_messages: list[object] | None = None
        self.ran = False

    async def run(self, input: RunAgentInput) -> AsyncIterator[BaseEvent]:
        """Snapshot the history-injected input messages, then emit the script in order."""
        self.ran = True
        self.seen_messages = list(input.messages)
        for event in self.events:
            yield event


class GatedAgent(AGUIAgent):
    """An inner agent that emits pre-events then blocks on a gate (for cancellation)."""

    def __init__(self, name: str, pre_events: list[BaseEvent], gate: asyncio.Event) -> None:
        super().__init__(name)
        self.pre_events = pre_events
        self.gate = gate

    async def run(self, input: RunAgentInput) -> AsyncIterator[BaseEvent]:
        """Emit the pre-events, then wait on the gate (cancelled while waiting)."""
        for event in self.pre_events:
            yield event
        await self.gate.wait()
        yield RunFinishedEvent(thread_id=input.thread_id, run_id=input.run_id)


class CrashingAgent(AGUIAgent):
    """An inner agent that emits pre-events then raises a raw exception mid-run."""

    def __init__(self, name: str, pre_events: list[BaseEvent], exc: Exception) -> None:
        super().__init__(name)
        self.pre_events = pre_events
        self.exc = exc

    async def run(self, input: RunAgentInput) -> AsyncIterator[BaseEvent]:
        """Emit the pre-events, then raise a raw exception (no terminal RunErrorEvent)."""
        for event in self.pre_events:
            yield event
        raise self.exc


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _unique(prefix: str = "") -> str:
    """Return a short unique string for test isolation."""
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def _input(thread_id: str, run_id: str, messages: list[object] | None = None) -> RunAgentInput:
    """Build a minimal :class:`RunAgentInput` for a run."""
    return RunAgentInput(
        thread_id=thread_id,
        run_id=run_id,
        state={},
        messages=messages or [],
        tools=[],
        context=[],
        forwarded_props={},
    )


def _repos(space: DRMemorySpace) -> tuple[ChatRepository, MessageRepository]:
    """Return a fresh ``(chat_repo, message_repo)`` pair sharing one cold registry.

    A new :class:`ChatSessionRegistry` means an empty cache, so every read-back
    resolves chats and messages straight from the live service (never a warm
    in-process cache) — proving the data really round-tripped.
    """
    registry = ChatSessionRegistry(space)
    return ChatRepository(space, registry), MessageRepository(space, registry)


def _full_turn(message_id: str) -> list[BaseEvent]:
    """Build a single assistant turn combining text, a tool call and reasoning."""
    return [
        RunStartedEvent(thread_id="t", run_id="r"),
        TextMessageStartEvent(message_id=message_id, role="assistant"),
        ToolCallStartEvent(
            tool_call_id="tc1", tool_call_name="search", parent_message_id=message_id
        ),
        ToolCallArgsEvent(tool_call_id="tc1", delta='{"q":'),
        ToolCallArgsEvent(tool_call_id="tc1", delta='"hi"}'),
        ToolCallEndEvent(tool_call_id="tc1"),
        ToolCallResultEvent(message_id=message_id, tool_call_id="tc1", content="result!"),
        ThinkingStartEvent(title="Planning"),
        ThinkingTextMessageStartEvent(),
        ThinkingTextMessageContentEvent(delta="think "),
        ThinkingTextMessageContentEvent(delta="hard"),
        ThinkingTextMessageEndEvent(),
        ThinkingEndEvent(),
        TextMessageContentEvent(message_id=message_id, delta="Here "),
        TextMessageContentEvent(message_id=message_id, delta="you go"),
        TextMessageEndEvent(message_id=message_id),
        RunFinishedEvent(thread_id="t", run_id="r"),
    ]


async def _drive(agent: AGUIStorageAgent, input: RunAgentInput) -> list[BaseEvent]:
    """Run *agent* to completion, returning every event it forwarded downstream."""
    return [event async for event in agent.run(input)]


async def _wait_until(pred: Callable[[], bool], *, timeout: float = 5.0) -> None:
    """Poll *pred* until it is true or *timeout* seconds elapse."""
    elapsed = 0.0
    while not pred():
        if elapsed >= timeout:
            raise AssertionError("condition was not met in time")
        await asyncio.sleep(0.02)
        elapsed += 0.02


async def _collect(handle: RunHandle) -> list[BaseEvent]:
    """Drain a :class:`RunHandle`'s client-facing event stream to the sentinel."""
    return [event async for event in handle.events()]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def client() -> DRMemoryServiceClient:  # type: ignore[return]
    """Return a ``DRMemoryServiceClient`` configured from environment variables."""
    async with DRMemoryServiceClient() as c:
        yield c


@pytest.fixture
async def space(client: DRMemoryServiceClient) -> DRMemorySpace:  # type: ignore[return]
    """Create a unique chat-history test memory space; delete after the test."""
    sp = await DRMemorySpace.post(
        client,
        description="Chat-history acceptance test space",
        deduplication_key=_unique("ch-space-"),
    )
    yield sp
    try:
        await sp.delete()
    except Exception:
        pass


# ── Scenario 1: full turn persisted as one event with typed nested models ──────


@skip_unless_live
async def test_full_turn_persists_one_event_with_typed_nested_models(
    space: DRMemorySpace,
) -> None:
    """GIVEN a scripted text + tool-call + reasoning turn
    WHEN driven through the storage agent against the live service
    THEN one message event persists and reconstructs typed nested models.
    """
    user_id = uuid.uuid4()
    thread_id = _unique("thread-")
    chat_repo, message_repo = _repos(space)
    events = _full_turn("m1")
    agent = AGUIStorageAgent(
        "assistant", user_id, chat_repo, message_repo, ScriptedAgent("inner", events)
    )

    emitted = await _drive(agent, _input(thread_id, _unique("run-")))

    # The wrapper forwards the inner stream verbatim and in order.
    assert [type(e) for e in emitted] == [type(e) for e in events]

    # Read back through a cold registry so the assertions hit the live service.
    read_chat_repo, read_message_repo = _repos(space)
    chat = await read_chat_repo.get_chat_by_thread_id(user_id, thread_id)
    assert chat is not None
    messages = list(await read_message_repo.get_chat_messages(chat.chat_uuid))

    assert len(messages) == 1
    message = messages[0]
    assert message.role == Role.ASSISTANT.value
    assert message.agui_id == "m1"
    assert message.content == "Here you go"
    assert message.in_progress is False
    assert message.status == MessageStatus.COMPLETE.value

    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert isinstance(tool_call, ToolCall)  # typed nested model, not a raw dict
    assert tool_call.name == "search"
    assert tool_call.arguments == '{"q":"hi"}'
    assert tool_call.content == "result!"
    assert tool_call.in_progress is False
    assert tool_call.status == MessageStatus.COMPLETE.value

    assert len(message.reasonings) == 1
    reasoning = message.reasonings[0]
    assert isinstance(reasoning, Reasoning)  # typed nested model, not a raw dict
    assert reasoning.name == "Planning"
    assert reasoning.content == "think hard"
    assert reasoning.in_progress is False


# ── Scenario 2: history injection across two turns ─────────────────────────────


@skip_unless_live
async def test_history_is_injected_across_two_turns(space: DRMemorySpace) -> None:
    """GIVEN a first turn stored under a thread
    WHEN a second turn runs on the same thread
    THEN the inner agent receives the prior turn translated into AG-UI history.
    """
    user_id = uuid.uuid4()
    thread_id = _unique("thread-")

    # Turn 1: an inbound user message plus an assistant reply carrying a tool call.
    turn1_events: list[BaseEvent] = [
        RunStartedEvent(thread_id="t", run_id="r1"),
        TextMessageStartEvent(message_id="a1", role="assistant"),
        ToolCallStartEvent(
            tool_call_id="tc1", tool_call_name="get_weather", parent_message_id="a1"
        ),
        ToolCallArgsEvent(tool_call_id="tc1", delta='{"city":"Paris"}'),
        ToolCallEndEvent(tool_call_id="tc1"),
        ToolCallResultEvent(message_id="a1", tool_call_id="tc1", content="sunny, 18C"),
        TextMessageContentEvent(message_id="a1", delta="It is sunny."),
        TextMessageEndEvent(message_id="a1"),
        RunFinishedEvent(thread_id="t", run_id="r1"),
    ]
    chat_repo1, message_repo1 = _repos(space)
    agent1 = AGUIStorageAgent(
        "assistant", user_id, chat_repo1, message_repo1, ScriptedAgent("inner-1", turn1_events)
    )
    await _drive(
        agent1,
        _input(thread_id, _unique("run-"), [UserMessage(id="u1", content="weather in Paris?")]),
    )

    # Turn 2: a fresh storage agent (cold repos) on the same thread.
    turn2_events: list[BaseEvent] = [
        RunStartedEvent(thread_id="t", run_id="r2"),
        TextMessageStartEvent(message_id="a2", role="assistant"),
        TextMessageContentEvent(message_id="a2", delta="Tomorrow too."),
        TextMessageEndEvent(message_id="a2"),
        RunFinishedEvent(thread_id="t", run_id="r2"),
    ]
    chat_repo2, message_repo2 = _repos(space)
    inner2 = ScriptedAgent("inner-2", turn2_events)
    agent2 = AGUIStorageAgent("assistant", user_id, chat_repo2, message_repo2, inner2)
    await _drive(
        agent2,
        _input(thread_id, _unique("run-"), [UserMessage(id="u2", content="and tomorrow?")]),
    )

    # The inner agent for turn 2 saw the full translated history: the first user
    # message, the assistant reply (carrying its tool call), the tool result, and
    # the freshly-persisted second user message — in that order.
    assert inner2.seen_messages is not None
    seen = inner2.seen_messages
    assert [m.role for m in seen] == [  # type: ignore[attr-defined]
        Role.USER.value,
        Role.ASSISTANT.value,
        Role.TOOL.value,
        Role.USER.value,
    ]
    assert [m.content for m in seen] == [  # type: ignore[attr-defined]
        "weather in Paris?",
        "It is sunny.",
        "sunny, 18C",
        "and tomorrow?",
    ]
    assistant_msg = seen[1]
    assert assistant_msg.tool_calls is not None  # type: ignore[attr-defined]
    assert assistant_msg.tool_calls[0].function.name == "get_weather"  # type: ignore[attr-defined]


# ── Scenario 3: idempotent chat create by (user, thread_id) ────────────────────


@skip_unless_live
async def test_chat_create_is_idempotent_per_user_and_thread(space: DRMemorySpace) -> None:
    """GIVEN a chat created for a ``(user, thread)``
    WHEN a second create (with a cold registry) uses the same pair
    THEN the existing chat is adopted and no duplicate session is created.
    """
    user_id = uuid.uuid4()
    thread_id = _unique("thread-")

    chat_repo1, _ = _repos(space)
    first = await chat_repo1.create_chat(
        ChatCreate(user_uuid=user_id, name="First", thread_id=thread_id)
    )

    # A cold registry forces the second create to rediscover the chat via the
    # live indexed lookup rather than an in-process cache.
    chat_repo2, _ = _repos(space)
    second = await chat_repo2.create_chat(
        ChatCreate(user_uuid=user_id, name="Second", thread_id=thread_id)
    )

    assert second.id == first.id
    assert second.chat_uuid == first.chat_uuid

    all_chats = await chat_repo2.get_all_chats(user_id)
    assert len([c for c in all_chats if c.thread_id == thread_id]) == 1


# ── Scenario 4: cancellation → interrupted ─────────────────────────────────────


@skip_unless_live
async def test_cancel_persists_interrupted_via_stream_manager(space: DRMemorySpace) -> None:
    """GIVEN a live run blocked mid-stream
    WHEN it is cancelled through the stream manager
    THEN the still-active record is finalised as ``interrupted``.
    """
    user_id = uuid.uuid4()
    thread_id = _unique("thread-")
    run_id = _unique("run-")
    chat_repo, message_repo = _repos(space)

    gate = asyncio.Event()
    pre: list[BaseEvent] = [
        RunStartedEvent(thread_id="t", run_id="r"),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="partial"),
    ]
    inner = GatedAgent("inner", pre, gate)
    manager = StreamPersistenceManager(
        lambda: AGUIStorageAgent("assistant", user_id, chat_repo, message_repo, inner)
    )

    handle = await manager.run(_input(thread_id, run_id))
    collected: list[BaseEvent] = []

    async def consume() -> None:
        async for event in handle.events():
            collected.append(event)

    reader = asyncio.create_task(consume())
    await _wait_until(lambda: len(collected) >= len(pre))

    assert handle.cancel() is True
    await handle.wait()
    await asyncio.wait_for(reader, timeout=5.0)  # events() terminated (never hangs)

    read_chat_repo, read_message_repo = _repos(space)
    chat = await read_chat_repo.get_chat_by_thread_id(user_id, thread_id)
    assert chat is not None
    messages = list(await read_message_repo.get_chat_messages(chat.chat_uuid))

    assert len(messages) == 1
    assert messages[0].content == "partial"
    assert messages[0].in_progress is False
    assert messages[0].status == MessageStatus.INTERRUPTED.value


# ── Scenario 5: disconnect survival ────────────────────────────────────────────


@skip_unless_live
async def test_disconnect_survival_drains_to_complete(space: DRMemorySpace) -> None:
    """GIVEN a client that never reads the event stream
    WHEN it only awaits the run's completion
    THEN persistence still drains the run to ``complete``.
    """
    user_id = uuid.uuid4()
    thread_id = _unique("thread-")
    chat_repo, message_repo = _repos(space)

    events: list[BaseEvent] = [
        RunStartedEvent(thread_id="t", run_id="r"),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="Hello "),
        TextMessageContentEvent(message_id="m1", delta="world"),
        TextMessageEndEvent(message_id="m1"),
        RunFinishedEvent(thread_id="t", run_id="r"),
    ]
    inner = ScriptedAgent("inner", events)
    manager = StreamPersistenceManager(
        lambda: AGUIStorageAgent("assistant", user_id, chat_repo, message_repo, inner)
    )

    handle = await manager.run(_input(thread_id, _unique("run-")))
    # Simulate a disconnect: never iterate handle.events(), just await the drain.
    await handle.wait()

    read_chat_repo, read_message_repo = _repos(space)
    chat = await read_chat_repo.get_chat_by_thread_id(user_id, thread_id)
    assert chat is not None
    messages = list(await read_message_repo.get_chat_messages(chat.chat_uuid))

    assert len(messages) == 1
    assert messages[0].content == "Hello world"
    assert messages[0].status == MessageStatus.COMPLETE.value


# ── Scenario 6: inner-agent crash → errored (distinct from cancel → interrupted) ─


@skip_unless_live
async def test_inner_crash_persists_errored_and_synthesizes_run_error(
    space: DRMemorySpace,
) -> None:
    """GIVEN an inner agent that raises a raw exception mid-run
    WHEN driven through the stream manager against the live service
    THEN a terminal RunError is synthesised AND the record is finalised ``errored``.
    """
    user_id = uuid.uuid4()
    thread_id = _unique("thread-")
    chat_repo, message_repo = _repos(space)

    pre: list[BaseEvent] = [
        RunStartedEvent(thread_id="t", run_id="r"),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="partial"),
    ]
    inner = CrashingAgent("inner", pre, RuntimeError("kaboom"))
    manager = StreamPersistenceManager(
        lambda: AGUIStorageAgent("assistant", user_id, chat_repo, message_repo, inner)
    )

    handle = await manager.run(_input(thread_id, _unique("run-")))
    # A hang would never terminate; the timeout is the assertion that it does not.
    emitted = await asyncio.wait_for(_collect(handle), timeout=10.0)
    await handle.wait()

    # The client stops waiting: a terminal RunErrorEvent is synthesised.
    assert isinstance(emitted[-1], RunErrorEvent)
    assert emitted[-1].code == ErrorCodes.INTERNAL_ERROR.value
    assert "kaboom" in emitted[-1].message

    # And the still-active record was finalised as ``errored`` (not ``interrupted``).
    read_chat_repo, read_message_repo = _repos(space)
    chat = await read_chat_repo.get_chat_by_thread_id(user_id, thread_id)
    assert chat is not None
    messages = list(await read_message_repo.get_chat_messages(chat.chat_uuid))

    assert len(messages) == 1
    assert messages[0].content == "partial"
    assert messages[0].in_progress is False
    assert messages[0].status == MessageStatus.ERRORED.value
    assert "kaboom" in (messages[0].error or "")
