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

"""AG-UI event emitter for CrewAI streaming.

Translates a CrewAI chunk/tool-event stream into a well-formed AG-UI event sequence.
The emitter owns the open step and the open text/reasoning message, so callers just
declare intent (``step``/``text``/``reasoning``/``tool_call``/``finish``) and every
opened step and message is closed before the next one opens.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

from ag_ui.core import EventType
from ag_ui.core import ReasoningEndEvent
from ag_ui.core import ReasoningMessageContentEvent
from ag_ui.core import ReasoningMessageEndEvent
from ag_ui.core import ReasoningMessageStartEvent
from ag_ui.core import ReasoningStartEvent
from ag_ui.core import StepFinishedEvent
from ag_ui.core import StepStartedEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallResultEvent
from ag_ui.core import ToolCallStartEvent

from datarobot_genai.crewai.streaming_events import ToolCallRecord


def _tool_call_record_to_events(record: ToolCallRecord, parent_message_id: str) -> list[Any]:
    """Convert a CrewAI tool-call record into its AG-UI ``ToolCall*`` event(s).

    ``parent_message_id`` is the assistant bubble the call follows, or "" when none was open.
    """
    if record.kind == "call":
        return [
            ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=record.tool_call_id,
                tool_call_name=record.name,
                parent_message_id=parent_message_id,
            ),
            ToolCallArgsEvent(
                type=EventType.TOOL_CALL_ARGS, tool_call_id=record.tool_call_id, delta=record.args
            ),
            ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id=record.tool_call_id),
        ]
    return [
        ToolCallResultEvent(
            type=EventType.TOOL_CALL_RESULT,
            message_id=record.tool_call_id,
            tool_call_id=record.tool_call_id,
            content=record.content,
            role="tool",
        )
    ]


class AGUIStreamEmitter:
    """Stateful translator from CrewAI stream events to AG-UI events.

    Owns the open step and the open text/reasoning message; every opened step/message is
    closed before the next opens. Methods yield bare AG-UI events -- the caller attaches
    usage metrics and the run-level ``RunStarted``/``RunFinished`` events.
    """

    def __init__(self) -> None:
        self._role = ""  # role of the open step ("" = none open)
        self._mid = str(uuid.uuid4())  # current message id
        self._text = False
        self._reasoning = False

    def step(self, role: str) -> Iterator[Any]:
        """Make ``role`` the open step, closing the previous one first. No-op if ``role`` is
        unchanged; an empty ``role`` only closes the current step.
        """
        if role == self._role:
            return
        if self._role:
            yield from self.close_messages()
            yield StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=self._role)
            self._mid = str(uuid.uuid4())
        if role:
            yield StepStartedEvent(type=EventType.STEP_STARTED, step_name=role)
        self._role = role

    def _reasoning_id(self) -> str:
        """Reasoning message id, derived from the text id but distinct from it: a UI grouping by id
        renders reasoning as its own block instead of folding it into the assistant bubble. Matches
        the langgraph/llamaindex adapters (``uuid5(NAMESPACE_OID, f"{text_id}-reasoning")``).
        """
        return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{self._mid}-reasoning"))

    def reasoning(self, on: bool) -> Iterator[Any]:
        """Open or close the reasoning message at a reasoning-mode boundary."""
        if on and not self._reasoning:
            yield ReasoningStartEvent(
                type=EventType.REASONING_START, message_id=self._reasoning_id()
            )
            yield ReasoningMessageStartEvent(
                type=EventType.REASONING_MESSAGE_START,
                message_id=self._reasoning_id(),
                role="reasoning",
            )
            self._reasoning = True
        elif not on and self._reasoning:
            yield from self._close_reasoning()

    def text(self, delta: str) -> Iterator[Any]:
        """Emit a content delta as reasoning content (in reasoning mode) or assistant text."""
        if self._reasoning:
            yield ReasoningMessageContentEvent(
                type=EventType.REASONING_MESSAGE_CONTENT,
                message_id=self._reasoning_id(),
                delta=delta,
            )
            return
        if not self._text:
            yield TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=self._mid)
            self._text = True
        yield TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT, message_id=self._mid, delta=delta
        )

    def tool_call(self, record: ToolCallRecord) -> Iterator[Any]:
        """Emit a tool call/result as a clean sibling: open its owning agent's step, close any
        open message, and attach the call to that message if one was open.
        """
        if record.kind == "call" and record.agent_role:
            yield from self.step(record.agent_role)
        parent = self._mid if (self._text or self._reasoning) else ""
        yield from self.close_messages()
        yield from _tool_call_record_to_events(record, parent)
        self._mid = str(uuid.uuid4())

    def finish(self) -> Iterator[Any]:
        """Close any open message and the open step at end of stream."""
        yield from self.close_messages()
        if self._role:
            yield StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=self._role)
            self._role = ""

    def close_messages(self) -> Iterator[Any]:
        """Close the open text/reasoning message, if any (at a step boundary or stream end)."""
        if self._text:
            yield TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=self._mid)
            self._text = False
        if self._reasoning:
            yield from self._close_reasoning()

    def _close_reasoning(self) -> Iterator[Any]:
        yield ReasoningMessageEndEvent(
            type=EventType.REASONING_MESSAGE_END, message_id=self._reasoning_id()
        )
        yield ReasoningEndEvent(type=EventType.REASONING_END, message_id=self._reasoning_id())
        self._reasoning = False
