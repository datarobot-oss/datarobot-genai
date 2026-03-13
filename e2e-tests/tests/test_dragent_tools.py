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

from __future__ import annotations

import os

import httpx
import pytest

from tests.conftest import GENERATE_STREAM_PATH
from tests.conftest import collect_ag_ui_events
from tests.conftest import collect_text
from tests.conftest import make_generate_payload
from tests.conftest import parse_sse_events

_is_nat = os.environ.get("DRAGENT_FRAMEWORK", "langgraph") == "nat"

CALCULATOR_PROMPT = (
    "You MUST use the calculator tool to compute the following expression. "
    "Do NOT compute it yourself. Call the calculator tool with this exact input: "
    "(1234 * 567890) + 91011. "
    "Report only the exact numeric result from the tool, no explanation, no formatting, no other text."
)

EXPECTED_RESULT = str((1234 * 567890) + 91011)


@pytest.fixture(scope="module")
def calculator_ag_ui_events(http_client: httpx.Client) -> list[dict]:  # type: ignore[type-arg]
    """Single LLM call shared by all calculator tests."""
    payload = make_generate_payload(CALCULATOR_PROMPT)
    with http_client.stream("POST", GENERATE_STREAM_PATH, json=payload) as response:
        assert response.status_code == 200
        sse_events = parse_sse_events(response)
    return collect_ag_ui_events(sse_events)


# Not all frameworks emit TOOL_CALL AG-UI events yet (CrewAI, LlamaIndex).
# NAT has no Python calculator tool - tool invocation is tested implicitly
# via the tool_calling_agent - chat_completion pipeline in test_dragent_streaming.py.
@pytest.mark.xfail(reason="Not all frameworks emit TOOL_CALL AG-UI events yet", strict=False)
def test_calculator_tool_is_called(calculator_ag_ui_events: list[dict]) -> None:  # type: ignore[type-arg]
    """Agent uses calculator tool when asked to compute."""
    event_types = {e["type"] for e in calculator_ag_ui_events}

    tool_types = {"TOOL_CALL_START", "TOOL_CALL_END", "TOOL_CALL_ARGS", "TOOL_CALL_RESULT"}
    assert event_types & tool_types, f"No tool call events found. Got: {event_types}"


# NAT agents don't have a Python calculator tool - their tool invocation is
# tested implicitly via the tool_calling_agent - chat_completion pipeline
# in the streaming test (test_dragent_streaming.py).
@pytest.mark.xfail(
    condition=_is_nat,
    reason="NAT has no Python calculator tool; tool invocation tested via streaming pipeline",
    strict=False,
)
def test_calculator_result_correct(calculator_ag_ui_events: list[dict]) -> None:  # type: ignore[type-arg]
    """Calculator tool returns the correct result."""
    full_text = collect_text(calculator_ag_ui_events)
    assert EXPECTED_RESULT in full_text, (
        f"Expected '{EXPECTED_RESULT}' in response. Got: {full_text[:500]}"
    )
