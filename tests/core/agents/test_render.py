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

"""Unit tests for :func:`datarobot_genai.core.agents.render.render_event`."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from ag_ui.core.events import EventType

from datarobot_genai.core.agents.render import TOOL_RESULT_MAX_LEN
from datarobot_genai.core.agents.render import render_event

_LOG = "datarobot_genai.core.agents.render"


@pytest.mark.parametrize(
    "event_type",
    (EventType.TEXT_MESSAGE_CONTENT, EventType.TEXT_MESSAGE_CHUNK),
)
def test_render_event_text_message_content_and_chunk(event_type: EventType) -> None:
    out = render_event(event_type, delta="hello")
    assert out is not None
    assert "hello" in out


def test_render_event_text_message_end() -> None:
    assert render_event(EventType.TEXT_MESSAGE_END) == "\n"


def test_render_event_text_message_start() -> None:
    assert render_event(EventType.TEXT_MESSAGE_START) is None


@pytest.mark.parametrize(
    "event_type",
    (EventType.REASONING_MESSAGE_CONTENT, EventType.REASONING_MESSAGE_CHUNK),
)
def test_render_event_reasoning_message_content_and_chunk(event_type: EventType) -> None:
    out = render_event(event_type, delta="think")
    assert out is not None
    assert "think" in out


@pytest.mark.parametrize(
    "event_type",
    (EventType.REASONING_END, EventType.REASONING_MESSAGE_END),
)
def test_render_event_reasoning_end_variants(event_type: EventType) -> None:
    assert render_event(event_type) == "\n"


@pytest.mark.parametrize(
    "event_type",
    (EventType.REASONING_START, EventType.REASONING_MESSAGE_START),
)
def test_render_event_reasoning_start_variants(event_type: EventType) -> None:
    assert render_event(event_type) is None


def test_render_event_tool_call_start() -> None:
    out = render_event(EventType.TOOL_CALL_START, name="search")
    assert out is not None
    assert "search" in out
    assert "Tool Call" in out


def test_render_event_tool_call_args() -> None:
    out = render_event(EventType.TOOL_CALL_ARGS, delta='{"q": 1}')
    assert out is not None
    assert '{"q": 1}' in out


def test_render_event_tool_call_end() -> None:
    assert render_event(EventType.TOOL_CALL_END) == "\n"


def test_render_event_tool_call_result() -> None:
    out = render_event(EventType.TOOL_CALL_RESULT, content="ok")
    assert out is not None
    assert "ok" in out


def test_render_event_tool_call_result_truncates() -> None:
    long_ = "z" * (TOOL_RESULT_MAX_LEN + 5)
    out = render_event(EventType.TOOL_CALL_RESULT, content=long_)
    assert out is not None
    assert "\u2026" in out
    assert long_ not in out


def test_render_event_step_started() -> None:
    out = render_event(EventType.STEP_STARTED, name="my_step")
    assert out is not None
    assert "my_step" in out


def test_render_event_step_finished() -> None:
    assert render_event(EventType.STEP_FINISHED) is None


def test_render_event_run_started() -> None:
    out = render_event(EventType.RUN_STARTED)
    assert out is not None
    assert "Run started" in out


def test_render_event_run_finished() -> None:
    out = render_event(EventType.RUN_FINISHED)
    assert out is not None
    assert "Run finished" in out


def test_render_event_run_error() -> None:
    out = render_event(EventType.RUN_ERROR, message="oops")
    assert out is not None
    assert "oops" in out


def test_render_event_custom_not_heartbeat() -> None:
    out = render_event(EventType.CUSTOM, name="MyEvent")
    assert out is not None
    assert "MyEvent" in out


def test_render_event_custom_heartbeat_suppresses_name() -> None:
    assert render_event(EventType.CUSTOM, name="Heartbeat") is None


def test_render_event_unhandled_event_type_in_enum_logs_and_returns_none(
    caplog: Any,
) -> None:
    with caplog.at_level(logging.DEBUG, logger=_LOG):
        out = render_event(EventType.TOOL_CALL_CHUNK)
    assert out is None
    messages = [r.getMessage() for r in caplog.records]
    assert any("TOOL_CALL_CHUNK" in m for m in messages)
    assert any("Unhandled" in m for m in messages)
