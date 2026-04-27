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
from colorama import Fore
from colorama import Style

from datarobot_genai.core.agents.render import TOOL_RESULT_MAX_LEN
from datarobot_genai.core.agents.render import render_event

_LOG = "datarobot_genai.core.agents.render"


@pytest.mark.parametrize(
    "event_type",
    (EventType.TEXT_MESSAGE_CONTENT, EventType.TEXT_MESSAGE_CHUNK),
)
def test_render_event_text_message_content_and_chunk(event_type: EventType) -> None:
    # GIVEN a text content or chunk event with a delta
    delta = "hello"

    # WHEN we render it
    out = render_event(event_type, delta=delta)

    # THEN the output is the delta wrapped in cyan styling
    expected = f"{Fore.CYAN}{delta}{Style.RESET_ALL}"
    assert out == expected


def test_render_event_text_message_end() -> None:
    # GIVEN a text message end event
    event_type = EventType.TEXT_MESSAGE_END

    # WHEN we render it
    out = render_event(event_type)

    # THEN the output is a single newline
    assert out == "\n"


def test_render_event_text_message_start() -> None:
    # GIVEN a text message start event
    event_type = EventType.TEXT_MESSAGE_START

    # WHEN we render it
    out = render_event(event_type)

    # THEN nothing is emitted
    assert out is None


@pytest.mark.parametrize(
    "event_type",
    (EventType.REASONING_MESSAGE_CONTENT, EventType.REASONING_MESSAGE_CHUNK),
)
def test_render_event_reasoning_message_content_and_chunk(event_type: EventType) -> None:
    # GIVEN a reasoning content or chunk event with a delta
    delta = "think"

    # WHEN we render it
    out = render_event(event_type, delta=delta)

    # THEN the output is the delta wrapped in yellow styling
    expected = f"{Fore.YELLOW}{delta}{Style.RESET_ALL}"
    assert out == expected


@pytest.mark.parametrize(
    "event_type",
    (EventType.REASONING_END, EventType.REASONING_MESSAGE_END),
)
def test_render_event_reasoning_end_variants(event_type: EventType) -> None:
    # GIVEN a reasoning-end event (parametrized variant)
    end_type = event_type

    # WHEN we render it
    out = render_event(end_type)

    # THEN the output is a single newline
    assert out == "\n"


@pytest.mark.parametrize(
    "event_type",
    (EventType.REASONING_START, EventType.REASONING_MESSAGE_START),
)
def test_render_event_reasoning_start_variants(event_type: EventType) -> None:
    # GIVEN a reasoning-start event (parametrized variant)
    start_type = event_type

    # WHEN we render it
    out = render_event(start_type)

    # THEN nothing is emitted
    assert out is None


def test_render_event_tool_call_start() -> None:
    # GIVEN a tool call start with a name
    name = "search"

    # WHEN we render it
    out = render_event(EventType.TOOL_CALL_START, name=name)

    # THEN the header and arguments label match the styled template
    expected = (
        f"\n{Fore.MAGENTA}\u25b6 Tool Call: {Fore.MAGENTA}{Style.DIM}{name}{Style.RESET_ALL}"
        f"\n{Fore.MAGENTA}  Arguments: {Style.RESET_ALL}"
    )
    assert out == expected


def test_render_event_tool_call_args() -> None:
    # GIVEN tool call args with a JSON delta
    delta = '{"q": 1}'

    # WHEN we render it
    out = render_event(EventType.TOOL_CALL_ARGS, delta=delta)

    # THEN the delta is dim magenta
    expected = f"{Fore.MAGENTA}{Style.DIM}{delta}{Style.RESET_ALL}"
    assert out == expected


def test_render_event_tool_call_end() -> None:
    # GIVEN a tool call end event
    event_type = EventType.TOOL_CALL_END

    # WHEN we render it
    out = render_event(event_type)

    # THEN the output is a single newline
    assert out == "\n"


def test_render_event_tool_call_result() -> None:
    # GIVEN a tool call result with content
    content = "ok"

    # WHEN we render it
    out = render_event(EventType.TOOL_CALL_RESULT, content=content)

    # THEN the result line includes the content and trailing newline
    expected = f"{Fore.MAGENTA}  Result: {Fore.MAGENTA}{Style.DIM}{content}{Style.RESET_ALL}\n"
    assert out == expected


def test_render_event_tool_call_result_truncates() -> None:
    # GIVEN tool result content longer than the max length
    long_content = "z" * (TOOL_RESULT_MAX_LEN + 5)
    truncated = "z" * TOOL_RESULT_MAX_LEN + "\u2026"

    # WHEN we render it
    out = render_event(EventType.TOOL_CALL_RESULT, content=long_content)

    # THEN the output uses the truncated prefix plus ellipsis
    expected = f"{Fore.MAGENTA}  Result: {Fore.MAGENTA}{Style.DIM}{truncated}{Style.RESET_ALL}\n"
    assert out == expected


def test_render_event_step_started() -> None:
    # GIVEN a step started event with a name
    name = "my_step"

    # WHEN we render it
    out = render_event(EventType.STEP_STARTED, name=name)

    # THEN the dim step line includes the name and newline
    expected = f"{Style.DIM}Step: {name}{Style.RESET_ALL}\n"
    assert out == expected


def test_render_event_step_finished() -> None:
    # GIVEN a step finished event
    event_type = EventType.STEP_FINISHED

    # WHEN we render it
    out = render_event(event_type)

    # THEN nothing is emitted
    assert out is None


def test_render_event_run_started() -> None:
    # GIVEN a run started event
    event_type = EventType.RUN_STARTED

    # WHEN we render it
    out = render_event(event_type)

    # THEN the dim run-started line ends with a newline
    expected = f"{Style.DIM}Run started{Style.RESET_ALL}\n"
    assert out == expected


def test_render_event_run_finished() -> None:
    # GIVEN a run finished event
    event_type = EventType.RUN_FINISHED

    # WHEN we render it
    out = render_event(event_type)

    # THEN the success line includes the checkmark and newlines
    expected = f"\n{Fore.GREEN}\u2705 Run finished.{Style.RESET_ALL}\n"
    assert out == expected


def test_render_event_run_error() -> None:
    # GIVEN a run error with a message
    message = "oops"

    # WHEN we render it
    out = render_event(EventType.RUN_ERROR, message=message)

    # THEN the failure line includes the message and newline
    expected = f"{Fore.RED}\u274c Run failed: {message}{Style.RESET_ALL}\n"
    assert out == expected


def test_render_event_custom_not_heartbeat() -> None:
    # GIVEN a custom event that is not Heartbeat
    name = "MyEvent"

    # WHEN we render it
    out = render_event(EventType.CUSTOM, name=name)

    # THEN the bracketed name line is dim and ends with newline
    expected = f"{Style.DIM}[{name}]{Style.RESET_ALL}\n"
    assert out == expected


def test_render_event_custom_heartbeat_suppresses_name() -> None:
    # GIVEN a custom Heartbeat event
    name = "Heartbeat"

    # WHEN we render it
    out = render_event(EventType.CUSTOM, name=name)

    # THEN nothing is emitted
    assert out is None


def test_render_event_unhandled_event_type_in_enum_logs_and_returns_none(
    caplog: Any,
) -> None:
    # GIVEN an enum value with no formatter branch and debug logging enabled
    event_type = EventType.TOOL_CALL_CHUNK

    # WHEN we render it
    with caplog.at_level(logging.DEBUG, logger=_LOG):
        out = render_event(event_type)

    # THEN there is no output string and one debug log names the event type
    assert out is None
    messages = [r.getMessage() for r in caplog.records]
    assert messages == ["Unhandled event type: EventType.TOOL_CALL_CHUNK"]
