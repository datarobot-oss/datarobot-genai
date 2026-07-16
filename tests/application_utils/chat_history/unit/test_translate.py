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

"""Unit tests for :func:`translate_messages` and :class:`ExtendedBaseMessage`.

The tests follow GIVEN-WHEN-THEN and cover the wire shape the AG-UI history
replay depends on: overall ordering by message ``timestamp``, the per-message
``message → tool results → reasonings`` fan-out, assistant tool-call attachment,
the tool-result id/tool_call_id column quirk, the empty-tool-content fallback,
and the extra bookkeeping fields carried through ``extra="allow"``.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime

from datarobot_genai.application_utils.chat_history.models import Message
from datarobot_genai.application_utils.chat_history.models import Reasoning
from datarobot_genai.application_utils.chat_history.models import Role
from datarobot_genai.application_utils.chat_history.models import ToolCall
from datarobot_genai.application_utils.chat_history.translate import ExtendedBaseMessage
from datarobot_genai.application_utils.chat_history.translate import translate_messages


def _ts(seconds: int) -> datetime:
    """Return a UTC datetime *seconds* past a fixed base (for deterministic ordering)."""
    return datetime(2026, 1, 1, tzinfo=UTC).replace(second=seconds)


def _message(**kwargs: object) -> Message:
    """Build a ``Message`` with the required base ORM fields defaulted."""
    kwargs.setdefault("content", "")
    kwargs.setdefault("emitter_type", "agent")
    return Message(**kwargs)  # type: ignore[arg-type]


def test_messages_are_sorted_by_timestamp() -> None:
    """GIVEN messages out of order WHEN translated THEN they come out timestamp-ascending."""
    late = _message(role=Role.USER.value, content="second", timestamp=_ts(20), agui_id="b")
    early = _message(role=Role.USER.value, content="first", timestamp=_ts(10), agui_id="a")

    out = list(translate_messages([late, early]))

    assert [m.content for m in out] == ["first", "second"]
    assert [m.id for m in out] == ["a", "b"]


def test_assistant_message_carries_openai_tool_calls_then_results() -> None:
    """GIVEN an assistant message with a tool call WHEN translated THEN a result follows it."""
    tool_call = ToolCall(
        agui_id="tc-agui",
        tool_call_id="tc-id",
        name="search",
        arguments='{"q": 1}',
        content="found it",
        created_at=_ts(5),
    )
    message = _message(
        role=Role.ASSISTANT.value,
        content="answer",
        name="assistant",
        agui_id="m1",
        timestamp=_ts(10),
        tool_calls=[tool_call],
    )

    out = list(translate_messages([message]))

    # The assistant message carries the OpenAI-style tool_calls entry.
    assistant = out[0]
    assert assistant.role == "assistant"
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0].id == "tc-agui"
    assert assistant.tool_calls[0].function.name == "search"
    assert assistant.tool_calls[0].function.arguments == '{"q": 1}'

    # The tool *result* follows, with the id/tool_call_id columns swapped.
    result = out[1]
    assert result.role == Role.TOOL.value
    assert result.id == "tc-id"  # public id is the tool_call_id
    assert result.tool_call_id == "tc-agui"  # tool_call_id carries the AG-UI id
    assert result.content == "found it"


def test_empty_tool_result_content_falls_back_to_completed_name() -> None:
    """GIVEN a tool call with no content WHEN translated THEN the result is 'Completed <name>'."""
    tool_call = ToolCall(
        agui_id="tc", tool_call_id="tc", name="lookup", content="", created_at=_ts(1)
    )
    message = _message(
        role=Role.ASSISTANT.value, agui_id="m", timestamp=_ts(2), tool_calls=[tool_call]
    )

    out = list(translate_messages([message]))

    assert out[1].content == "Completed lookup"


def test_tool_calls_and_reasonings_sorted_by_created_at() -> None:
    """GIVEN nested items out of order WHEN translated THEN each group is created_at-ascending."""
    tc_late = ToolCall(agui_id="tc2", tool_call_id="tc2", name="b", created_at=_ts(20))
    tc_early = ToolCall(agui_id="tc1", tool_call_id="tc1", name="a", created_at=_ts(10))
    r_late = Reasoning(agui_id="r2", name="think-2", content="later", created_at=_ts(40))
    r_early = Reasoning(agui_id="r1", name="think-1", content="earlier", created_at=_ts(30))
    message = _message(
        role=Role.ASSISTANT.value,
        agui_id="m",
        timestamp=_ts(1),
        tool_calls=[tc_late, tc_early],
        reasonings=[r_late, r_early],
    )

    out = list(translate_messages([message]))

    # out[0] is the message; [1],[2] tool results; [3],[4] reasonings.
    assert [m.id for m in out[1:3]] == ["tc1", "tc2"]
    assert [m.role for m in out[3:5]] == [Role.REASONING.value, Role.REASONING.value]
    assert [m.content for m in out[3:5]] == ["earlier", "later"]


def test_non_assistant_message_has_no_tool_calls_attached() -> None:
    """GIVEN a user message WHEN translated THEN it is a plain message with no tool_calls."""
    message = _message(role=Role.USER.value, content="hi", agui_id="u1", timestamp=_ts(1))

    out = list(translate_messages([message]))

    assert len(out) == 1
    assert out[0].role == Role.USER.value
    assert out[0].tool_calls is None
    assert out[0].uuid == str(message.message_uuid)


def test_extended_fields_are_carried_through() -> None:
    """GIVEN a message with status/error WHEN translated THEN the extra fields survive."""
    message = _message(
        role=Role.ASSISTANT.value,
        content="done",
        agui_id="m",
        timestamp=_ts(1),
        in_progress=False,
        status="complete",
        error=None,
    )

    out = list(translate_messages([message]))

    assert isinstance(out[0], ExtendedBaseMessage)
    assert out[0].in_progress is False
    assert out[0].status == "complete"
    assert out[0].error is None


def test_id_falls_back_to_uuid_when_agui_id_missing() -> None:
    """GIVEN a message without an agui_id WHEN translated THEN id falls back to the uuid."""
    message = _message(role=Role.USER.value, content="hi", agui_id=None, timestamp=_ts(1))

    out = list(translate_messages([message]))

    assert out[0].id == str(message.message_uuid)
