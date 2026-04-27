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
    assert render_event(event_type, delta="hello") == f"{Fore.CYAN}hello{Style.RESET_ALL}"


def test_render_event_text_message_end() -> None:
    assert render_event(EventType.TEXT_MESSAGE_END) == "\n"


def test_render_event_text_message_start() -> None:
    assert render_event(EventType.TEXT_MESSAGE_START) is None


@pytest.mark.parametrize(
    "event_type",
    (EventType.REASONING_MESSAGE_CONTENT, EventType.REASONING_MESSAGE_CHUNK),
)
def test_render_event_reasoning_message_content_and_chunk(event_type: EventType) -> None:
    assert render_event(event_type, delta="think") == f"{Fore.YELLOW}think{Style.RESET_ALL}"


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
    assert render_event(EventType.TOOL_CALL_START, name="search") == (
        f"\n{Fore.MAGENTA}\u25b6 Tool Call: {Fore.MAGENTA}{Style.DIM}search{Style.RESET_ALL}"
        f"\n{Fore.MAGENTA}  Arguments: {Style.RESET_ALL}"
    )


def test_render_event_tool_call_args() -> None:
    delta = '{"q": 1}'
    assert render_event(EventType.TOOL_CALL_ARGS, delta=delta) == (
        f"{Fore.MAGENTA}{Style.DIM}{delta}{Style.RESET_ALL}"
    )


def test_render_event_tool_call_end() -> None:
    assert render_event(EventType.TOOL_CALL_END) == "\n"


def test_render_event_tool_call_result() -> None:
    assert render_event(EventType.TOOL_CALL_RESULT, content="ok") == (
        f"{Fore.MAGENTA}  Result: {Fore.MAGENTA}{Style.DIM}ok{Style.RESET_ALL}\n"
    )


def test_render_event_tool_call_result_truncates() -> None:
    long_ = "z" * (TOOL_RESULT_MAX_LEN + 5)
    truncated = "z" * TOOL_RESULT_MAX_LEN + "\u2026"
    assert render_event(EventType.TOOL_CALL_RESULT, content=long_) == (
        f"{Fore.MAGENTA}  Result: {Fore.MAGENTA}{Style.DIM}{truncated}{Style.RESET_ALL}\n"
    )


def test_render_event_step_started() -> None:
    assert render_event(EventType.STEP_STARTED, name="my_step") == (
        f"{Style.DIM}Step: my_step{Style.RESET_ALL}\n"
    )


def test_render_event_step_finished() -> None:
    assert render_event(EventType.STEP_FINISHED) is None


def test_render_event_run_started() -> None:
    assert render_event(EventType.RUN_STARTED) == f"{Style.DIM}Run started{Style.RESET_ALL}\n"


def test_render_event_run_finished() -> None:
    assert render_event(EventType.RUN_FINISHED) == (
        f"\n{Fore.GREEN}\u2705 Run finished.{Style.RESET_ALL}\n"
    )


def test_render_event_run_error() -> None:
    assert render_event(EventType.RUN_ERROR, message="oops") == (
        f"{Fore.RED}\u274c Run failed: oops{Style.RESET_ALL}\n"
    )


def test_render_event_custom_not_heartbeat() -> None:
    assert render_event(EventType.CUSTOM, name="MyEvent") == (
        f"{Style.DIM}[MyEvent]{Style.RESET_ALL}\n"
    )


def test_render_event_custom_heartbeat_suppresses_name() -> None:
    assert render_event(EventType.CUSTOM, name="Heartbeat") is None


def test_render_event_unhandled_event_type_in_enum_logs_and_returns_none(
    caplog: Any,
) -> None:
    with caplog.at_level(logging.DEBUG, logger=_LOG):
        out = render_event(EventType.TOOL_CALL_CHUNK)
    assert out is None
    assert [r.getMessage() for r in caplog.records] == [
        "Unhandled event type: EventType.TOOL_CALL_CHUNK",
    ]
