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

"""Unit tests for :class:`AGUIStorageAgent`, the AG-UI → storage state machine.

Tests drive scripted AG-UI event streams through the agent against the
dependency-free in-memory repositories and assert the persisted result.  They
cover text / tool-call / reasoning / run-lifecycle / step handling, content
buffering + flush, one-event-per-message folding, inbound-user persistence with
both ``INVALID_INPUT`` rejections, history injection, cancellation → interrupted,
and all four extensibility points (build hooks, dispatch registry, category
handler override, and loose repository coupling).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from collections.abc import Callable
from typing import Any
from uuid import UUID
from uuid import uuid4

import pytest
from ag_ui.core import AssistantMessage
from ag_ui.core import BaseEvent
from ag_ui.core import CustomEvent
from ag_ui.core import ReasoningEncryptedValueEvent
from ag_ui.core import ReasoningEndEvent
from ag_ui.core import ReasoningMessageChunkEvent
from ag_ui.core import ReasoningMessageContentEvent
from ag_ui.core import ReasoningMessageEndEvent
from ag_ui.core import ReasoningMessageStartEvent
from ag_ui.core import ReasoningStartEvent
from ag_ui.core import RunAgentInput
from ag_ui.core import RunErrorEvent
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import StepFinishedEvent
from ag_ui.core import StepStartedEvent
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

from datarobot_genai.application_utils.chat_history.ag_ui_storage import AGUIAgent
from datarobot_genai.application_utils.chat_history.ag_ui_storage import AGUIStorageAgent
from datarobot_genai.application_utils.chat_history.ag_ui_storage import ErrorCodes
from datarobot_genai.application_utils.chat_history.ag_ui_storage import StorageState
from datarobot_genai.application_utils.chat_history.models import Chat
from datarobot_genai.application_utils.chat_history.models import Message
from datarobot_genai.application_utils.chat_history.models import MessageCreate
from datarobot_genai.application_utils.chat_history.models import MessageStatus
from datarobot_genai.application_utils.chat_history.models import MessageUpdate
from datarobot_genai.application_utils.chat_history.repositories import ChatRepositoryLike
from datarobot_genai.application_utils.chat_history.repositories import MessageRepositoryLike
from tests.application_utils.chat_history.unit.conftest import RichMessage
from tests.application_utils.chat_history.unit.fake_repos import InMemoryChatRepository
from tests.application_utils.chat_history.unit.fake_repos import InMemoryMessageRepository
from tests.application_utils.chat_history.unit.fake_repos import RecordingMessageRepository

THREAD = "thread-1"


# ── Scripted inner agents ──────────────────────────────────────────────────────


class ScriptedAgent(AGUIAgent):
    """An inner agent that emits a fixed list of events and records its input."""

    def __init__(self, name: str, events: list[BaseEvent]) -> None:
        super().__init__(name)
        self.events = events
        self.seen_messages: list[Any] | None = None
        self.ran = False

    async def run(self, input: RunAgentInput) -> AsyncIterator[BaseEvent]:
        """Snapshot the (history-injected) input messages, then emit the script."""
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
        yield _run_finished()  # pragma: no cover - never reached in the cancel test


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


# ── Builders / helpers ──────────────────────────────────────────────────────────


def _run_started() -> RunStartedEvent:
    return RunStartedEvent(thread_id=THREAD, run_id="run-1")


def _run_finished() -> RunFinishedEvent:
    return RunFinishedEvent(thread_id=THREAD, run_id="run-1")


def _input(messages: list[Any] | None = None, thread_id: str = THREAD) -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
        run_id="run-1",
        state={},
        messages=messages or [],
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


async def _drive(
    events: list[BaseEvent],
    *,
    inbound: list[Any] | None = None,
    chat_repo: ChatRepositoryLike | None = None,
    message_repo: MessageRepositoryLike | None = None,
    agent_cls: type[AGUIStorageAgent] = AGUIStorageAgent,
    user_id: UUID | None = None,
    **kwargs: Any,
) -> tuple[AGUIStorageAgent, ScriptedAgent, Any, Any, list[BaseEvent]]:
    """Run a scripted stream through the storage agent to completion.

    Returns the storage agent, the inner agent, both repositories and the list of
    events the storage agent emitted downstream.
    """
    chat_repo = chat_repo or InMemoryChatRepository()
    message_repo = message_repo or InMemoryMessageRepository()
    inner = ScriptedAgent("inner", events)
    agent = agent_cls("assistant", user_id or uuid4(), chat_repo, message_repo, inner, **kwargs)
    emitted = [event async for event in agent.run(_input(inbound))]
    return agent, inner, chat_repo, message_repo, emitted


async def _wait_until(pred: Callable[[], bool], *, timeout: float = 2.0) -> None:
    elapsed = 0.0
    while not pred():
        if elapsed >= timeout:
            raise AssertionError("condition was not met in time")
        await asyncio.sleep(0.01)
        elapsed += 0.01


def _messages(repo: InMemoryMessageRepository) -> list[Message]:
    return list(repo.messages.values())


# ── Text messages ───────────────────────────────────────────────────────────────


async def test_text_turn_persists_one_completed_message() -> None:
    """GIVEN a text turn WHEN driven THEN one complete message with joined content persists."""
    _, _, _, message_repo, _ = await _drive(_text_turn())

    messages = _messages(message_repo)
    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].agui_id == "m1"
    assert messages[0].content == "Hello world"
    assert messages[0].in_progress is False
    assert messages[0].status == MessageStatus.COMPLETE.value


async def test_events_are_forwarded_downstream_unchanged() -> None:
    """GIVEN a stream WHEN driven THEN every inner event is yielded downstream in order."""
    events = _text_turn()
    _, _, _, _, emitted = await _drive(events)

    assert [type(e) for e in emitted] == [type(e) for e in events]


# ── Buffering ────────────────────────────────────────────────────────────────────


async def test_low_threshold_flushes_more_often_than_high_threshold() -> None:
    """GIVEN content deltas WHEN the flush threshold is low THEN more writes occur, same result."""
    deltas = ("aaaaa", "bbbbb")

    _, _, _, repo_hi, _ = await _drive(
        _text_turn(deltas=deltas),
        message_repo=RecordingMessageRepository(),
        minimal_chunk_to_persist=10_000,
    )
    _, _, _, repo_lo, _ = await _drive(
        _text_turn(deltas=deltas),
        message_repo=RecordingMessageRepository(),
        minimal_chunk_to_persist=1,
    )

    assert repo_lo.update_message_calls > repo_hi.update_message_calls
    assert _messages(repo_hi)[0].content == "aaaaabbbbb"
    assert _messages(repo_lo)[0].content == "aaaaabbbbb"


# ── One-event-per-message folding ────────────────────────────────────────────────


async def test_two_message_ids_fold_into_two_messages() -> None:
    """GIVEN two distinct message ids WHEN driven THEN exactly two messages persist."""
    events = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="first"),
        TextMessageEndEvent(message_id="m1"),
        TextMessageStartEvent(message_id="m2", role="assistant"),
        TextMessageContentEvent(message_id="m2", delta="second"),
        TextMessageEndEvent(message_id="m2"),
        _run_finished(),
    ]

    _, _, _, message_repo, _ = await _drive(events)

    messages = _messages(message_repo)
    assert len(messages) == 2
    assert {m.agui_id: m.content for m in messages} == {"m1": "first", "m2": "second"}


async def test_new_agui_id_closes_out_the_prior_unfinished_message() -> None:
    """GIVEN a new message id before the prior ends WHEN driven THEN the prior is closed out."""
    events = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="partial"),
        TextMessageStartEvent(message_id="m2", role="assistant"),
        TextMessageContentEvent(message_id="m2", delta="done"),
        TextMessageEndEvent(message_id="m2"),
        _run_finished(),
    ]

    _, _, _, message_repo, _ = await _drive(events)

    messages = {m.agui_id: m for m in _messages(message_repo)}
    assert len(messages) == 2
    assert messages["m1"].content == "partial"
    assert messages["m1"].in_progress is False  # closed out on the new id
    assert messages["m2"].content == "done"


# ── Tool calls ────────────────────────────────────────────────────────────────────


async def test_tool_call_is_nested_in_the_message_body() -> None:
    """GIVEN a tool-call sub-stream WHEN driven THEN a completed nested tool call persists."""
    events = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        ToolCallStartEvent(tool_call_id="tc1", tool_call_name="search", parent_message_id="m1"),
        ToolCallArgsEvent(tool_call_id="tc1", delta='{"q":'),
        ToolCallArgsEvent(tool_call_id="tc1", delta='"hi"}'),
        ToolCallEndEvent(tool_call_id="tc1"),
        ToolCallResultEvent(message_id="m1", tool_call_id="tc1", content="result!"),
        TextMessageEndEvent(message_id="m1"),
        _run_finished(),
    ]

    _, _, _, message_repo, _ = await _drive(events)

    messages = _messages(message_repo)
    assert len(messages) == 1
    tool_calls = messages[0].tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "search"
    assert tool_calls[0].arguments == '{"q":"hi"}'
    assert tool_calls[0].content == "result!"
    assert tool_calls[0].in_progress is False
    assert tool_calls[0].status == MessageStatus.COMPLETE.value


# ── Reasoning ─────────────────────────────────────────────────────────────────────


async def test_reasoning_is_nested_and_named_from_the_thinking_title() -> None:
    """GIVEN a reasoning sub-stream WHEN driven THEN the reasoning is named from the title."""
    events = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        ThinkingStartEvent(title="Planning"),
        ThinkingTextMessageStartEvent(),
        ThinkingTextMessageContentEvent(delta="step 1 "),
        ThinkingTextMessageContentEvent(delta="step 2"),
        ThinkingTextMessageEndEvent(),
        ThinkingEndEvent(),
        TextMessageEndEvent(message_id="m1"),
        _run_finished(),
    ]

    _, _, _, message_repo, _ = await _drive(events)

    reasonings = _messages(message_repo)[0].reasonings
    assert len(reasonings) == 1
    assert reasonings[0].name == "Planning"
    assert reasonings[0].content == "step 1 step 2"
    assert reasonings[0].in_progress is False
    # A normally-ended reasoning step is COMPLETE, not stranded at "active".
    assert reasonings[0].status == MessageStatus.COMPLETE.value


async def test_reasoning_content_without_start_uses_title_not_object() -> None:
    """GIVEN reasoning content with no start WHEN driven THEN the name is the title string."""
    events = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        ThinkingStartEvent(title="Reasoning about X"),
        # No ThinkingTextMessageStartEvent — the content event must create the reasoning.
        ThinkingTextMessageContentEvent(delta="because"),
        ThinkingTextMessageEndEvent(),
        ThinkingEndEvent(),
        TextMessageEndEvent(message_id="m1"),
        _run_finished(),
    ]

    _, _, _, message_repo, _ = await _drive(events)

    reasonings = _messages(message_repo)[0].reasonings
    assert len(reasonings) == 1
    # The storage.py:520 bug set name to the MessageReasoning object; it must be the title.
    assert reasonings[0].name == "Reasoning about X"
    assert reasonings[0].content == "because"
    assert reasonings[0].status == MessageStatus.COMPLETE.value


async def test_reasoning_events_persist_nested_reasoning_with_agui_id() -> None:
    """GIVEN a current-protocol Reasoning* sub-stream WHEN driven THEN it persists like Thinking."""
    events = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        ReasoningStartEvent(message_id="r1"),
        ReasoningMessageStartEvent(message_id="r1", role="reasoning"),
        ReasoningMessageContentEvent(message_id="r1", delta="step 1 "),
        ReasoningMessageContentEvent(message_id="r1", delta="step 2"),
        ReasoningMessageEndEvent(message_id="r1"),
        ReasoningEndEvent(message_id="r1"),
        TextMessageEndEvent(message_id="m1"),
        _run_finished(),
    ]

    _, _, _, message_repo, _ = await _drive(events)

    reasonings = _messages(message_repo)[0].reasonings
    assert len(reasonings) == 1
    assert reasonings[0].content == "step 1 step 2"
    # The Reasoning* events carry a message_id, persisted as agui_id (Thinking* had none).
    assert reasonings[0].agui_id == "r1"
    assert reasonings[0].in_progress is False
    assert reasonings[0].status == MessageStatus.COMPLETE.value


async def test_reasoning_content_without_message_start_creates_reasoning() -> None:
    """GIVEN Reasoning content with no message-start WHEN driven THEN the content creates it."""
    events = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        ReasoningStartEvent(message_id="r1"),
        # No ReasoningMessageStartEvent — the content event must create the reasoning.
        ReasoningMessageContentEvent(message_id="r1", delta="because"),
        ReasoningMessageEndEvent(message_id="r1"),
        ReasoningEndEvent(message_id="r1"),
        TextMessageEndEvent(message_id="m1"),
        _run_finished(),
    ]

    _, _, _, message_repo, _ = await _drive(events)

    reasonings = _messages(message_repo)[0].reasonings
    assert len(reasonings) == 1
    assert reasonings[0].content == "because"
    assert reasonings[0].agui_id == "r1"
    assert reasonings[0].in_progress is False
    assert reasonings[0].status == MessageStatus.COMPLETE.value


async def test_reasoning_message_chunk_folds_content_and_finalizes_at_run_finish() -> None:
    """GIVEN self-contained ReasoningMessageChunk events WHEN driven THEN content is folded."""
    events = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        ReasoningMessageChunkEvent(message_id="r1", delta="chunk a "),
        ReasoningMessageChunkEvent(message_id="r1", delta="chunk b"),
        TextMessageEndEvent(message_id="m1"),
        _run_finished(),
    ]

    _, _, _, message_repo, _ = await _drive(events)

    reasonings = _messages(message_repo)[0].reasonings
    assert len(reasonings) == 1
    assert reasonings[0].content == "chunk a chunk b"
    assert reasonings[0].agui_id == "r1"
    # No explicit reasoning end; the run-finish finalisation completes the step.
    assert reasonings[0].in_progress is False
    assert reasonings[0].status == MessageStatus.COMPLETE.value


async def test_reasoning_encrypted_value_event_is_ignored() -> None:
    """GIVEN a ReasoningEncryptedValue event WHEN driven THEN nothing is persisted for it."""
    events: list[BaseEvent] = [
        _run_started(),
        ReasoningEncryptedValueEvent(
            subtype="message", entity_id="e1", encrypted_value="opaque-blob"
        ),
        _run_finished(),
    ]

    _, _, _, message_repo, _ = await _drive(events)

    # The opaque encrypted value has no cleartext home and is not chat history.
    assert _messages(message_repo) == []


async def test_reasoning_before_text_start_folds_into_the_same_message() -> None:
    """GIVEN reasoning before the answer's TextMessageStart WHEN driven THEN one message persists.

    The reasoning opens an anonymous assistant message; when the answer's real
    message id arrives it is adopted in place, so the reasoning and the answer
    text persist on a single message instead of two.
    """
    events = [
        _run_started(),
        # Reasoning arrives before any TextMessageStart (typical of reasoning models).
        ReasoningStartEvent(message_id="r1"),
        ReasoningMessageStartEvent(message_id="r1", role="reasoning"),
        ReasoningMessageContentEvent(message_id="r1", delta="thinking"),
        ReasoningMessageEndEvent(message_id="r1"),
        ReasoningEndEvent(message_id="r1"),
        # The answer's message id must be adopted by the reasoning-opened message.
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="answer"),
        TextMessageEndEvent(message_id="m1"),
        _run_finished(),
    ]

    _, _, _, message_repo, _ = await _drive(events)

    messages = _messages(message_repo)
    assert len(messages) == 1
    message = messages[0]
    assert message.agui_id == "m1"  # the anonymous message adopted the answer id
    assert message.content == "answer"
    assert len(message.reasonings) == 1
    assert message.reasonings[0].content == "thinking"
    assert message.reasonings[0].agui_id == "r1"


async def test_thinking_before_text_start_folds_into_the_same_message() -> None:
    """GIVEN deprecated Thinking events before the TextMessageStart WHEN driven THEN one message."""
    events = [
        _run_started(),
        ThinkingStartEvent(title="Planning"),
        ThinkingTextMessageStartEvent(),
        ThinkingTextMessageContentEvent(delta="step 1"),
        ThinkingTextMessageEndEvent(),
        ThinkingEndEvent(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="answer"),
        TextMessageEndEvent(message_id="m1"),
        _run_finished(),
    ]

    _, _, _, message_repo, _ = await _drive(events)

    messages = _messages(message_repo)
    assert len(messages) == 1
    assert messages[0].agui_id == "m1"
    assert messages[0].content == "answer"
    assert [r.name for r in messages[0].reasonings] == ["Planning"]
    assert messages[0].reasonings[0].content == "step 1"


# ── Run lifecycle / steps ─────────────────────────────────────────────────────────


async def test_run_error_marks_the_active_message_errored() -> None:
    """GIVEN a RunError mid-message WHEN driven THEN the message is errored with a coded message."""
    events = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="partial"),
        RunErrorEvent(message="boom", code="INTERNAL_ERROR"),
    ]

    _, _, _, message_repo, _ = await _drive(events)

    message = _messages(message_repo)[0]
    assert message.content == "partial"
    assert message.in_progress is False
    assert message.status == MessageStatus.ERRORED.value
    assert message.error == "[INTERNAL_ERROR] boom"


async def test_active_step_is_recorded_on_the_created_message() -> None:
    """GIVEN a step around a message WHEN driven THEN the message records the step name."""
    events = [
        _run_started(),
        StepStartedEvent(step_name="research"),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="x"),
        TextMessageEndEvent(message_id="m1"),
        StepFinishedEvent(step_name="research"),
        _run_finished(),
    ]

    _, _, _, message_repo, _ = await _drive(events)

    assert _messages(message_repo)[0].step == "research"


# ── Inbound user persistence + history injection ──────────────────────────────────


async def test_inbound_user_message_is_persisted_and_history_injected() -> None:
    """GIVEN an inbound user message WHEN driven THEN it persists and is replayed as history."""
    inbound = [UserMessage(id="u1", content="What is 2+2?")]
    events = [
        _run_started(),
        TextMessageStartEvent(message_id="a1", role="assistant"),
        TextMessageContentEvent(message_id="a1", delta="4"),
        TextMessageEndEvent(message_id="a1"),
        _run_finished(),
    ]

    _, inner, chat_repo, message_repo, _ = await _drive(events, inbound=inbound)

    messages = _messages(message_repo)
    by_role = {m.role: m for m in messages}
    assert by_role["user"].content == "What is 2+2?"
    assert by_role["user"].agui_id == "u1"
    assert by_role["user"].in_progress is False
    assert by_role["assistant"].content == "4"

    # The inner agent saw the persisted user message replayed as history.
    assert inner.seen_messages is not None
    assert [m.content for m in inner.seen_messages] == ["What is 2+2?"]

    # And the chat name was derived from the first user message.
    chat = (await chat_repo.get_all_chats(None))[0]
    assert chat.name == "What is 2+2?"


async def test_new_non_user_message_is_rejected_with_invalid_input() -> None:
    """GIVEN an inbound non-user message WHEN driven THEN the run stops with INVALID_INPUT."""
    inbound = [AssistantMessage(id="a1", content="I should not be created")]

    _, inner, _, message_repo, emitted = await _drive(_text_turn(), inbound=inbound)

    assert len(emitted) == 1
    assert isinstance(emitted[0], RunErrorEvent)
    assert emitted[0].code == ErrorCodes.INVALID_INPUT.value
    assert inner.ran is False  # the inner agent never ran
    assert _messages(message_repo) == []


async def test_message_from_another_chat_is_rejected_with_invalid_input() -> None:
    """GIVEN an inbound message owned by another chat WHEN driven THEN INVALID_INPUT halts it."""

    class CrossChatMessageRepo(InMemoryMessageRepository):
        async def get_message_by_agui_id(self, chat_id: UUID, agui_id: str) -> Message | None:
            # Simulate a message that exists but belongs to a *different* chat.
            return Message(
                content="x",
                emitter_type="user",
                chat_id=uuid4(),
                agui_id=agui_id,
                role="user",
            )

    inbound = [UserMessage(id="dup", content="hi")]
    _, inner, _, _, emitted = await _drive(
        _text_turn(), inbound=inbound, message_repo=CrossChatMessageRepo()
    )

    assert len(emitted) == 1
    assert isinstance(emitted[0], RunErrorEvent)
    assert emitted[0].code == ErrorCodes.INVALID_INPUT.value
    assert "same chat" in emitted[0].message
    assert inner.ran is False


# ── Cancellation ──────────────────────────────────────────────────────────────────


async def test_cancellation_finalizes_in_progress_records_as_interrupted() -> None:
    """GIVEN a cancelled run WHEN finalised THEN still-active records become interrupted."""
    gate = asyncio.Event()
    pre: list[BaseEvent] = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="partial"),
    ]
    inner = GatedAgent("inner", pre, gate)
    message_repo = InMemoryMessageRepository()
    agent = AGUIStorageAgent("assistant", uuid4(), InMemoryChatRepository(), message_repo, inner)

    collected: list[BaseEvent] = []

    async def consume() -> None:
        async for event in agent.run(_input()):
            collected.append(event)

    task = asyncio.create_task(consume())
    await _wait_until(lambda: len(collected) >= len(pre))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    messages = _messages(message_repo)
    assert len(messages) == 1
    assert messages[0].content == "partial"
    assert messages[0].in_progress is False
    assert messages[0].status == MessageStatus.INTERRUPTED.value


# ── Inner-agent crash (raw exception → errored) ───────────────────────────────────


async def test_inner_agent_crash_finalizes_in_progress_message_as_errored() -> None:
    """GIVEN an inner agent that raises mid-message WHEN driven THEN it re-raises and errors it."""
    pre: list[BaseEvent] = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        TextMessageContentEvent(message_id="m1", delta="partial"),
    ]
    inner = CrashingAgent("inner", pre, RuntimeError("boom"))
    message_repo = InMemoryMessageRepository()
    agent = AGUIStorageAgent("assistant", uuid4(), InMemoryChatRepository(), message_repo, inner)

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in agent.run(_input()):
            pass

    messages = _messages(message_repo)
    assert len(messages) == 1
    assert messages[0].content == "partial"
    assert messages[0].in_progress is False
    assert messages[0].status == MessageStatus.ERRORED.value
    assert "boom" in (messages[0].error or "")


async def test_inner_agent_crash_finalizes_in_progress_tool_call_as_errored() -> None:
    """GIVEN a crash mid tool-call WHEN driven THEN the nested tool call is errored too."""
    pre: list[BaseEvent] = [
        _run_started(),
        TextMessageStartEvent(message_id="m1", role="assistant"),
        ToolCallStartEvent(tool_call_id="tc1", tool_call_name="search", parent_message_id="m1"),
        ToolCallArgsEvent(tool_call_id="tc1", delta='{"q":"hi"}'),
    ]
    inner = CrashingAgent("inner", pre, RuntimeError("boom"))
    message_repo = InMemoryMessageRepository()
    agent = AGUIStorageAgent("assistant", uuid4(), InMemoryChatRepository(), message_repo, inner)

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in agent.run(_input()):
            pass

    message = _messages(message_repo)[0]
    assert message.in_progress is False
    assert message.status == MessageStatus.ERRORED.value
    assert "boom" in (message.error or "")

    tool_calls = message.tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0].arguments == '{"q":"hi"}'
    assert tool_calls[0].in_progress is False
    assert tool_calls[0].status == MessageStatus.ERRORED.value
    assert "boom" in (tool_calls[0].error or "")


# ── Extensibility point 1: message-build hook (custom field) ──────────────────────


class TaggedMessageCreate(MessageCreate):
    """A ``MessageCreate`` carrying an extra ``priority`` field."""

    priority: int = 0


class TaggingAgent(AGUIStorageAgent):
    """A storage agent that populates a custom ``priority`` via the build hook."""

    def build_message_create(
        self, state: StorageState, chat: Chat, agui_id: str | None, role: str | None
    ) -> MessageCreate:
        base = super().build_message_create(state, chat, agui_id, role)
        return TaggedMessageCreate(**base.model_dump(), priority=7)


class TaggedMessageRepository(InMemoryMessageRepository):
    """A repository that persists the extra ``priority`` onto a ``RichMessage``."""

    message_cls = RichMessage

    def build_message(self, data: MessageCreate) -> Message:
        message = super().build_message(data)
        message.priority = getattr(data, "priority", 0)  # type: ignore[attr-defined]
        return message


async def test_build_hook_persists_a_custom_field() -> None:
    """GIVEN a build-hook override WHEN driven THEN the custom field is persisted."""
    _, _, _, message_repo, _ = await _drive(
        _text_turn(), agent_cls=TaggingAgent, message_repo=TaggedMessageRepository()
    )

    message = _messages(message_repo)[0]
    assert isinstance(message, RichMessage)
    assert message.priority == 7


# ── Extensibility point 2: dispatch registry (new event type) ─────────────────────


class CustomEventAgent(AGUIStorageAgent):
    """A storage agent that handles the AG-UI ``CustomEvent`` type."""

    @classmethod
    def event_handlers(cls) -> dict[type[BaseEvent], str]:
        return {**super().event_handlers(), CustomEvent: "handle_custom"}

    async def handle_custom(self, state: StorageState, chat: Chat, event: BaseEvent) -> None:
        assert isinstance(event, CustomEvent)
        await self._message_repo.create_message(
            MessageCreate(
                chat_id=chat.chat_uuid,
                role="assistant",
                agui_id=event.name,
                content=str(event.value),
                in_progress=False,
                status=MessageStatus.COMPLETE.value,
            )
        )


async def test_registered_custom_event_is_handled() -> None:
    """GIVEN a registered custom event WHEN driven THEN its handler persists a message."""
    events: list[BaseEvent] = [
        _run_started(),
        CustomEvent(name="note", value={"k": 1}),
        _run_finished(),
    ]

    _, _, _, message_repo, _ = await _drive(events, agent_cls=CustomEventAgent)

    messages = _messages(message_repo)
    assert len(messages) == 1
    assert messages[0].agui_id == "note"
    assert messages[0].content == "{'k': 1}"


async def test_unregistered_event_is_ignored_by_default() -> None:
    """GIVEN an unregistered event WHEN driven by the base agent THEN it is ignored."""
    events: list[BaseEvent] = [
        _run_started(),
        CustomEvent(name="note", value=1),
        _run_finished(),
    ]

    _, _, _, message_repo, _ = await _drive(events)

    assert _messages(message_repo) == []


# ── Extensibility point 3: category handler override ──────────────────────────────


class UppercasingAgent(AGUIStorageAgent):
    """A storage agent that transforms text content instead of storing it verbatim."""

    async def handle_text_message(self, state: StorageState, chat: Chat, event: BaseEvent) -> None:
        if isinstance(event, TextMessageStartEvent):
            await self._ensure_message_exists(state, chat, event.message_id, event.role)
        elif isinstance(event, TextMessageContentEvent):
            await self._ensure_message_exists(state, chat, event.message_id, None)
            assert state.active_message is not None
            state.active_message.content += event.delta.upper()
            await self._message_repo.update_message(
                state.active_message.message_uuid,
                MessageUpdate(content=state.active_message.content),
            )
        elif isinstance(event, TextMessageEndEvent):
            await self._ensure_message_exists(state, chat, event.message_id, None)
            assert state.active_message is not None
            await self._message_repo.update_message(
                state.active_message.message_uuid, MessageUpdate(in_progress=False)
            )


async def test_text_handler_override_transforms_content() -> None:
    """GIVEN a text-handler override WHEN driven THEN content is stored transformed."""
    _, _, _, message_repo, _ = await _drive(
        _text_turn(deltas=("hello ", "world")), agent_cls=UppercasingAgent
    )

    assert _messages(message_repo)[0].content == "HELLO WORLD"


# ── Extensibility point 4: loose repository coupling ──────────────────────────────


async def test_runs_against_protocol_conforming_fakes() -> None:
    """GIVEN dependency-free fakes WHEN used THEN they satisfy the protocols and drive a run."""
    chat_repo = InMemoryChatRepository()
    message_repo = InMemoryMessageRepository()
    assert isinstance(chat_repo, ChatRepositoryLike)
    assert isinstance(message_repo, MessageRepositoryLike)

    _, _, _, driven_repo, _ = await _drive(
        _text_turn(), chat_repo=chat_repo, message_repo=message_repo
    )

    assert _messages(driven_repo)[0].content == "Hello world"


# ── Guard rails ────────────────────────────────────────────────────────────────────


def test_cannot_wrap_a_storage_agent_in_another() -> None:
    """GIVEN a storage agent as inner WHEN wrapping THEN a ValueError is raised."""
    inner = AGUIStorageAgent(
        "x", uuid4(), InMemoryChatRepository(), InMemoryMessageRepository(), ScriptedAgent("i", [])
    )
    with pytest.raises(ValueError, match="second storage layer"):
        AGUIStorageAgent("y", uuid4(), InMemoryChatRepository(), InMemoryMessageRepository(), inner)
