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

import os

import httpx
import pytest
from ag_ui.core import EventType
from ag_ui.verify import validate_sequence

from dragent_tests.helpers import FRAMEWORK
from dragent_tests.helpers import FRAMEWORK_SUPPORTS_TOOL_CALLS
from dragent_tests.helpers import GENERATE_STREAM_PATH
from dragent_tests.helpers import collect_ag_ui_events
from dragent_tests.helpers import make_generate_payload
from dragent_tests.helpers import parse_sse_responses

pytestmark = pytest.mark.skipif(
    not os.environ.get("MCP_DEPLOYMENT_ID"),
    reason="MCP_DEPLOYMENT_ID not set; skipping MCP tests",
)

MCP_TOOL_PROMPT = (
    "You MUST use the search_datarobot_agentic_docs tool to search for 'MCP server'. "
    "Call it with query='MCP server' and max_results=1. "
    "Report only the title of the first result."
)

EXPECTED_TOOL_CALL_NAMES = {
    "search_datarobot_agentic_docs",
    "mcp_tools__search_datarobot_agentic_docs"
}

@pytest.mark.skipif(
    FRAMEWORK == "base",
    reason="Base framework does not implement anything, skipping MCP tool call tests",
)
def test_mcp_tool_is_called(http_client: httpx.Client) -> None:  # type: ignore[type-arg]
    """Agent invokes an MCP tool (search_datarobot_agentic_docs)."""
    # GIVEN: a prompt that invokes the search_datarobot_agentic_docs tool
    # WHEN: the agent is invoked
    payload = make_generate_payload(MCP_TOOL_PROMPT)
    with http_client.stream("POST", GENERATE_STREAM_PATH, json=payload) as response:
        assert response.status_code == 200
        sse_events = parse_sse_responses(response)

    # THEN: a response is correct AG UI events
    mcp_ag_ui_events = collect_ag_ui_events(sse_events)

    # THEN: a response is a valid AG-UI sequence
    validate_sequence(mcp_ag_ui_events)

    # THEN: the events contain tool call events (if framework supports tool calls)
    tool_types = {
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_END,
        EventType.TOOL_CALL_ARGS,
        EventType.TOOL_CALL_RESULT,
    }
    event_types = {e.type for e in mcp_ag_ui_events}
    if FRAMEWORK_SUPPORTS_TOOL_CALLS:
        assert event_types & tool_types, f"No tool call events found. Got: {event_types}"

        # THEN: the tool call events contain mcp_tools__search_datarobot_agentic_docs
        tool_call_names = {
            e.tool_call_name for e in mcp_ag_ui_events if e.type == EventType.TOOL_CALL_START
        }
        assert EXPECTED_TOOL_CALL_NAMES & tool_call_names, (
            "No tool call event found for mcp_tools__search_datarobot_agentic_docs. "
            f"Got: {tool_call_names}"
        )
    else:
        assert not event_types & tool_types, (
            f"Tool call events found when framework does not support them. Got: {event_types}"
        )
