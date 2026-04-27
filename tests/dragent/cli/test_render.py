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

"""Tests for :mod:`datarobot_genai.dragent.cli.render` (SSE and object adapters)."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

from ag_ui.core.events import EventType
from ag_ui.core.events import ReasoningMessageChunkEvent
from ag_ui.core.events import ReasoningMessageContentEvent
from ag_ui.core.events import RunErrorEvent
from ag_ui.core.events import RunFinishedEvent
from ag_ui.core.events import TextMessageChunkEvent
from ag_ui.core.events import TextMessageContentEvent
from ag_ui.core.events import ToolCallStartEvent

from datarobot_genai.dragent.cli.render import render_object_event
from datarobot_genai.dragent.cli.render import render_sse_event

_LOG = "datarobot_genai.dragent.cli.render"

# ---------------------------------------------------------------------------
# render_sse_event
# ---------------------------------------------------------------------------


@patch("datarobot_genai.dragent.cli.render.click.echo")
def test_render_sse_event_invalid_type_string_logs_and_does_not_echo(
    mock_echo: Any, caplog: Any
) -> None:
    # GIVEN a non-empty ``type`` that is not a valid :class:`EventType` value
    with caplog.at_level(logging.DEBUG, logger=_LOG):
        out = render_sse_event({"type": "NOT_A_VALID_EVENT_TYPE", "delta": "x"})
    assert out is None
    assert mock_echo.call_count == 0
    assert any("Unknown SSE event type" in r.message for r in caplog.records)


@patch("datarobot_genai.dragent.cli.render.click.echo")
def test_render_sse_event_empty_type_logs_and_does_not_echo(mock_echo: Any, caplog: Any) -> None:
    # GIVEN an event dict with no/empty type
    with caplog.at_level(logging.DEBUG, logger=_LOG):
        out = render_sse_event({})
    # WHEN/THEN: no terminal output; helper returns None; empty type is logged
    assert out is None
    assert mock_echo.call_count == 0
    assert any("Unhandled SSE event type" in r.message for r in caplog.records)


@patch("datarobot_genai.dragent.cli.render.click.echo")
def test_render_sse_event_run_error_returns_message_and_skips_render(
    mock_echo: Any,
) -> None:
    # GIVEN RUN_ERROR: remote layer raises; no styled echo
    out = render_sse_event(
        {"type": EventType.RUN_ERROR.value, "message": "stream failed"},
    )
    assert out == "stream failed"
    assert mock_echo.call_count == 0


@patch("datarobot_genai.dragent.cli.render.click.echo")
def test_render_sse_event_run_error_default_message(mock_echo: Any) -> None:
    out = render_sse_event({"type": EventType.RUN_ERROR.value})
    assert out == "Unknown error"
    assert mock_echo.call_count == 0


@patch("datarobot_genai.dragent.cli.render.click.echo")
def test_render_sse_event_renders_other_types_via_render_event_and_echos(
    mock_echo: Any,
) -> None:
    # GIVEN a normal (non-ERROR) SSE event
    out = render_sse_event(
        {
            "type": EventType.TEXT_MESSAGE_CONTENT.value,
            "delta": "hello",
        }
    )
    # THEN: click.echo is used (all streams unified to stdout) and the helper returns None
    assert out is None
    assert mock_echo.call_count == 1
    assert mock_echo.call_args[1].get("nl") is False
    assert "hello" in mock_echo.call_args[0][0]


@patch("datarobot_genai.dragent.cli.render.click.echo")
def test_render_sse_event_resolves_name_from_alternate_keys(mock_echo: Any) -> None:
    # GIVEN step metadata under ``name`` or ``step_name`` only
    render_sse_event(
        {
            "type": EventType.STEP_STARTED.value,
            "name": "via_name",
        }
    )
    assert "via_name" in mock_echo.call_args[0][0]

    mock_echo.reset_mock()
    render_sse_event(
        {
            "type": EventType.STEP_STARTED.value,
            "step_name": "via_step",
        }
    )
    assert "via_step" in mock_echo.call_args[0][0]

    mock_echo.reset_mock()
    render_sse_event(
        {
            "type": EventType.TOOL_CALL_START.value,
            "tool_call_name": "tool_fn",
        }
    )
    assert "tool_fn" in mock_echo.call_args[0][0]


# ---------------------------------------------------------------------------
# render_object_event
# ---------------------------------------------------------------------------


@patch("datarobot_genai.dragent.cli.render.click.echo")
def test_render_object_event_run_finished_does_not_echo(
    mock_echo: Any,
) -> None:
    # GIVEN: console prints its own "run finished" line
    ev = RunFinishedEvent(
        thread_id="t1",
        run_id="r1",
    )
    assert render_object_event(ev) is False
    assert mock_echo.call_count == 0


@patch("datarobot_genai.dragent.cli.render.click.echo")
def test_render_object_event_run_error_does_not_echo(
    mock_echo: Any,
) -> None:
    ev = RunErrorEvent(
        message="bad",
    )
    assert render_object_event(ev) is False
    assert mock_echo.call_count == 0


@patch("datarobot_genai.dragent.cli.render.click.echo")
def test_render_object_event_text_and_reasoning_delta_returns_true(
    mock_echo: Any,
) -> None:
    for ev in (
        TextMessageContentEvent(message_id="m1", delta="a"),
        TextMessageChunkEvent(message_id="m2", delta="b"),
        ReasoningMessageContentEvent(message_id="m3", delta="c"),
        ReasoningMessageChunkEvent(message_id="m4", delta="d"),
    ):
        mock_echo.reset_mock()
        assert render_object_event(ev) is True, type(ev)
        assert mock_echo.call_count == 1


@patch("datarobot_genai.dragent.cli.render.click.echo")
def test_render_object_event_text_chunk_empty_delta_returns_false(
    mock_echo: Any,
) -> None:
    # GIVEN chunk with no delta: still may paint empty styled span; return is False
    ev = TextMessageChunkEvent(
        message_id="m1",
        delta="",
    )
    assert render_object_event(ev) is False


@patch("datarobot_genai.dragent.cli.render.click.echo")
def test_render_object_event_tool_work_prints_false_for_streaming_flag(
    mock_echo: Any,
) -> None:
    # GIVEN a tool call start: we print, but this is not "streaming text" for the console flag
    ev = ToolCallStartEvent(
        tool_call_id="c1",
        tool_call_name="x",
    )
    assert render_object_event(ev) is False
    assert mock_echo.call_count == 1
