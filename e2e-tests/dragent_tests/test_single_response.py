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
from ag_ui.verify import validate_sequence
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse

from dragent_tests.helpers import AGENT
from dragent_tests.helpers import ALL_TEST_CASES
from dragent_tests.helpers import GENERATE_PATH
from dragent_tests.helpers import collect_text
from dragent_tests.helpers import make_generate_payload

if AGENT == "nat":
    pytest.skip(
        "NAT returns single response in chat completions format, and we do not yet care to fix it.", allow_module_level=True)

if not ALL_TEST_CASES:
    pytest.skip("Running minimal test set for non-LLM Gateway LLM, skipping single response tests", allow_module_level=True)


def test_generate_single(http_client: httpx.Client) -> None:
    """Concatenated text deltas produce a non-empty response."""
    # GIVEN: a payload that requests "Say 'hello world' and nothing else."
    payload = make_generate_payload("Say 'hello world' and nothing else.")

    # WHEN: called a generate endpoint for a single response (non-streaming)
    response = http_client.post(GENERATE_PATH, json=payload)
    assert response.status_code == 200
    # THEN: the response is non-streaming JSON
    assert "application/json" in response.headers.get("content-type", "")
    # THEN: the response is DRAgentEventResponse
    response_data = DRAgentEventResponse.model_validate_json(response.text)

    # THEN: the response contains AG-UI events
    assert len(response_data.events) > 0

    # THEN: the events are a valid AG-UI sequence
    validate_sequence(response_data.events)

    # THEN: there are events with text
    full_text = collect_text(response_data.events)
    assert len(full_text) > 0, "Expected non-empty text response"
