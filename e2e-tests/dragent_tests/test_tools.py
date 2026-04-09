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

import httpx
import pytest
from ag_ui.core import EventType
from ag_ui.verify import validate_sequence

from dragent_tests.helpers import AGENT
from dragent_tests.helpers import AGENT_SUPPORTS_TOOL_CALLS
from dragent_tests.helpers import AGENT_SUPPORTS_TOOL_CALLS_STREAMING
from dragent_tests.helpers import GENERATE_STREAM_PATH
from dragent_tests.helpers import LLM
from dragent_tests.helpers import collect_ag_ui_events
from dragent_tests.helpers import collect_text
from dragent_tests.helpers import make_generate_payload
from dragent_tests.helpers import parse_sse_responses

if not AGENT_SUPPORTS_TOOL_CALLS:
    pytest.skip(f"{AGENT} agent does not support tool calls, skipping tool call tests", allow_module_level=True)

if AGENT == "crewai" and LLM == "external":
    pytest.skip("BUZZOK-30223: CrewAI has issues with passing MCP tools to external LLM", allow_module_level=True)

GENERATE_OBJECTID_PROMPT = (
    "You MUST use the generate_objectid tool to generate an object ID for a deployment. "
    "Do NOT generate it yourself. Call the generate_objectid tool with this exact input: "
    "deployment. "
    "You MUST return the exact object ID from the tool, no explanation, "
    "no formatting, no other text. Do not write anything else. Always return the "
    "final answer to the user"
)

EXPECTED_GENERATE_OBJECTID_RESULT = "69cbb73789723b6936c6c9e1"

def test_generate_objectid_tool_is_called(http_client: httpx.Client) -> None:  # type: ignore[type-arg]
    """Agent uses calculator tool when asked to compute."""
    # GIVEN: a payload that requests the generate_objectid tool to generate an object ID for a
    # deployment
    payload = make_generate_payload(GENERATE_OBJECTID_PROMPT)
    # WHEN: the payload is streamed to the generate endpoint
    with http_client.stream("POST", GENERATE_STREAM_PATH, json=payload) as response:
        assert response.status_code == 200
        # THEN: the response is a valid AG-UI response
        sse_events = parse_sse_responses(response)

    # THEN: the response contains AG-UI events
    ag_ui_events = collect_ag_ui_events(sse_events)

    # THEN: the events are a valid AG-UI sequence
    validate_sequence(ag_ui_events)
    # THEN: there are events with tool call
    event_types = {e.type for e in ag_ui_events}
    tool_types = {
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_END,
        EventType.TOOL_CALL_ARGS,
        EventType.TOOL_CALL_RESULT,
    }
    if AGENT_SUPPORTS_TOOL_CALLS_STREAMING:
        assert event_types & tool_types, f"No tool call events found. Got: {event_types}"

        # THEN: the tool call events contain the generate_objectid tool
        tool_call_names = {
            e.tool_call_name for e in ag_ui_events if e.type == EventType.TOOL_CALL_START
        }
        assert "generate_objectid" in tool_call_names, (
            "No tool call event found for generate_objectid. "
            f"Got: {tool_call_names}"
        )
    else:
        assert not event_types & tool_types, (
            f"Tool call events found when framework does not support them. Got: {event_types}"
        )

    # THEN: the tool call events contain the correct result
    full_text = collect_text(ag_ui_events)
    assert EXPECTED_GENERATE_OBJECTID_RESULT in full_text, (
        f"Expected '{EXPECTED_GENERATE_OBJECTID_RESULT}' in response. Got: {full_text[:500]}"
    )


CALCULATOR_PROMPT = (
    "You MUST use the calculator tool to compute the following expression. "
    "Do NOT compute it yourself. Call the calculator tool with this exact input: "
    "(1234 * 567890) + 91011. "
    "You MUST return the exact numeric result from the tool, no explanation, "
    "no formatting, no other text. Do not write anything else. Always return the "
    "final answer to the user"
)

EXPECTED_CALCULATOR_RESULT = str((1234 * 567890) + 91011)

@pytest.mark.skip(
    "Obsolete test: I kept it here only to because it used to be a problem for crewai, "
    "but now it is fixed. We really don't want to run 2 tests on every run. "
)
def test_calculator_tool_is_called(http_client: httpx.Client) -> None:  # type: ignore[type-arg]
    """Agent uses calculator tool when asked to compute."""
    # GIVEN: a payload that requests the generate_objectid tool to generate an object ID for a
    # deployment
    payload = make_generate_payload(CALCULATOR_PROMPT)
    # WHEN: the payload is streamed to the generate endpoint
    with http_client.stream("POST", GENERATE_STREAM_PATH, json=payload) as response:
        assert response.status_code == 200
        # THEN: the response is a valid AG-UI response
        sse_events = parse_sse_responses(response)

    # THEN: the response contains AG-UI events
    ag_ui_events = collect_ag_ui_events(sse_events)

    # THEN: the events are a valid AG-UI sequence
    validate_sequence(ag_ui_events)

    # THEN: there are events with tool call
    event_types = {e.type for e in ag_ui_events}
    tool_types = {
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_END,
        EventType.TOOL_CALL_ARGS,
        EventType.TOOL_CALL_RESULT,
    }
    if AGENT_SUPPORTS_TOOL_CALLS:
        assert event_types & tool_types, f"No tool call events found. Got: {event_types}"

        # THEN: the tool call events contain the generate_objectid tool
        tool_call_names = {
            e.tool_call_name for e in ag_ui_events if e.type == EventType.TOOL_CALL_START
        }
        assert "calculator" in tool_call_names, (
            "No tool call event found for calculator. "
            f"Got: {tool_call_names}"
        )
    else:
        assert not event_types & tool_types, (
            f"Tool call events found when framework does not support them. Got: {event_types}"
        )

    # THEN: the tool call events contain the correct result
    full_text = collect_text(ag_ui_events)
    assert EXPECTED_CALCULATOR_RESULT in full_text, (
        f"Expected '{EXPECTED_CALCULATOR_RESULT}' in response. Got: {full_text[:500]}"
    )
