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
from ag_ui.verify import validate_sequence

from dragent_tests.helpers import GENERATE_STREAM_PATH
from dragent_tests.helpers import collect_ag_ui_events
from dragent_tests.helpers import collect_text
from dragent_tests.helpers import make_generate_payload
from dragent_tests.helpers import parse_sse_responses

if os.environ.get("AGENT") != "router":
    pytest.skip("Router e2e tests require AGENT=router", allow_module_level=True)


def test_router_streaming_ag_ui(http_client: httpx.Client) -> None:
    """Streaming /generate/stream returns a valid non-empty AG-UI sequence."""
    # GIVEN: a user message for the router-backed agent
    payload = make_generate_payload("Say 'hello' and nothing else.")

    # WHEN: the client streams the request to /generate/stream
    with http_client.stream("POST", GENERATE_STREAM_PATH, json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        sse_responses = parse_sse_responses(response)

    # THEN: SSE payloads decode to a valid AG-UI event sequence with non-empty text
    ag_ui_events = collect_ag_ui_events(sse_responses)
    validate_sequence(ag_ui_events)
    full_text = collect_text(ag_ui_events)
    assert len(full_text) > 0, "Expected non-empty text response"
