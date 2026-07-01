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

"""Unit tests for AGUIStreamEmitter -- the CrewAI -> AG-UI event state machine."""

from __future__ import annotations

from typing import Any

from ag_ui.core import EventType

from datarobot_genai.crewai.agui_stream import AGUIStreamEmitter
from datarobot_genai.crewai.streaming_events import ToolCallRecord


def _collect(*iterables: Any) -> list[Any]:
    events: list[Any] = []
    for it in iterables:
        events.extend(it)
    return events


def _types(events: list[Any]) -> list[EventType]:
    return [e.type for e in events]


def _call(tool_call_id: str, role: str) -> ToolCallRecord:
    return ToolCallRecord(
        kind="call", tool_call_id=tool_call_id, name="search", args="{}", agent_role=role
    )


def test_step_text_finish_is_well_formed() -> None:
    em = AGUIStreamEmitter()
    events = _collect(em.step("Planner"), em.text("hi"), em.finish())
    assert _types(events) == [
        EventType.STEP_STARTED,
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
        EventType.STEP_FINISHED,
    ]


def test_step_is_noop_when_role_unchanged() -> None:
    em = AGUIStreamEmitter()
    list(em.step("Planner"))
    assert list(em.step("Planner")) == []


def test_step_transition_closes_previous_message_then_step() -> None:
    em = AGUIStreamEmitter()
    events = _collect(em.step("Planner"), em.text("x"), em.step("Writer"))
    assert _types(events) == [
        EventType.STEP_STARTED,  # Planner
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,  # close Planner's message
        EventType.STEP_FINISHED,  # Planner
        EventType.STEP_STARTED,  # Writer
    ]


def test_reasoning_opens_content_and_closes() -> None:
    em = AGUIStreamEmitter()
    events = _collect(em.reasoning(True), em.text("thinking"), em.reasoning(False))
    assert _types(events) == [
        EventType.REASONING_START,
        EventType.REASONING_MESSAGE_START,
        EventType.REASONING_MESSAGE_CONTENT,
        EventType.REASONING_MESSAGE_END,
        EventType.REASONING_END,
    ]


def test_tool_call_same_role_closes_open_text_and_attaches() -> None:
    em = AGUIStreamEmitter()
    list(em.step("Planner"))
    text = list(em.text("partial"))
    mid = text[0].message_id

    events = list(em.tool_call(_call("t1", "Planner")))
    assert _types(events) == [
        EventType.TEXT_MESSAGE_END,  # close the open bubble first
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_ARGS,
        EventType.TOOL_CALL_END,
    ]
    start = events[1]
    assert start.parent_message_id == mid  # attached to the bubble it followed


def test_tool_call_different_role_transitions_step() -> None:
    em = AGUIStreamEmitter()
    list(em.step("Planner"))
    events = list(em.tool_call(_call("t1", "Writer")))
    assert _types(events)[:2] == [EventType.STEP_FINISHED, EventType.STEP_STARTED]
    assert EventType.TOOL_CALL_START in _types(events)


def test_tool_call_parent_empty_when_no_open_message() -> None:
    em = AGUIStreamEmitter()
    list(em.step("Planner"))
    events = list(em.tool_call(_call("t1", "Planner")))
    start = next(e for e in events if e.type == EventType.TOOL_CALL_START)
    assert start.parent_message_id == ""


def test_tool_result_emits_single_result_event() -> None:
    em = AGUIStreamEmitter()
    record = ToolCallRecord(kind="result", tool_call_id="t1", content="42")
    events = list(em.tool_call(record))
    assert _types(events) == [EventType.TOOL_CALL_RESULT]
    assert events[0].content == "42"
    assert events[0].tool_call_id == "t1"
