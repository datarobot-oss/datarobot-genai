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
from tests.conftest import make_generate_payload
from tests.conftest import parse_sse_events

pytestmark = pytest.mark.skipif(
    not os.environ.get("MCP_DEPLOYMENT_ID"),
    reason="MCP_DEPLOYMENT_ID not set; skipping MCP tests",
)


MCP_TOOL_PROMPT = (
    "You MUST use the search_datarobot_agentic_docs tool to search for 'MCP server'. "
    "Call it with query='MCP server' and max_results=1. "
    "Report only the title of the first result."
)


@pytest.fixture(scope="module")
def mcp_ag_ui_events(mcp_http_client: httpx.Client) -> list[dict]:  # type: ignore[type-arg]
    """Single MCP streaming call shared by all MCP tests."""
    payload = make_generate_payload(MCP_TOOL_PROMPT)
    with mcp_http_client.stream("POST", GENERATE_STREAM_PATH, json=payload) as response:
        assert response.status_code == 200
        sse_events = parse_sse_events(response)
    return collect_ag_ui_events(sse_events)


@pytest.mark.xfail(
    reason="NAT profiler callback deepcopy bug: copy.deepcopy(inputs) fails on MCP tool inputs "
    "containing coroutines/contextvars.",
    strict=False,
)
def test_mcp_tool_is_called(mcp_ag_ui_events: list[dict]) -> None:  # type: ignore[type-arg]
    """Agent invokes an MCP tool (search_datarobot_agentic_docs)."""
    event_types = {e["type"] for e in mcp_ag_ui_events}
    tool_types = {"TOOL_CALL_START", "TOOL_CALL_END", "TOOL_CALL_ARGS", "TOOL_CALL_RESULT"}
    assert event_types & tool_types, f"No tool call events found. Got: {event_types}"
