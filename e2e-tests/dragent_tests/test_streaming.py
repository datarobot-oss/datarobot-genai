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

from dragent_tests.helpers import GENERATE_STREAM_PATH
from dragent_tests.helpers import collect_ag_ui_events
from dragent_tests.helpers import collect_text
from dragent_tests.helpers import make_generate_payload
from dragent_tests.helpers import parse_sse_responses


# TODO: analyze that its a correct AG-UI sequence
def test_generate_streaming_produces_text(http_client: httpx.Client) -> None:
    """Concatenated text deltas produce a non-empty response."""
    # GIVEN: a payload that requests "Say 'hello world' and nothing else."
    payload = make_generate_payload("Say 'hello world' and nothing else.")

    # WHEN: the payload is streamed to the generate endpoint
    response = http_client.post(GENERATE_STREAM_PATH, json=payload)
    assert response.status_code == 200
    # THEN: the response is streaming
    assert "text/event-stream" in response.headers.get("content-type", "")
    # THEN: the response is a valid AG-UI response
    sse_responses = parse_sse_responses(response)

    # THEN: the response contains AG-UI events
    ag_ui_events = collect_ag_ui_events(sse_responses)

    # THEN: there are events with text
    full_text = collect_text(ag_ui_events)
    assert len(full_text) > 0, "Expected non-empty text response"
