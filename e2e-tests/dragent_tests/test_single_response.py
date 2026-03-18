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
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse

from dragent_tests.helpers import FRAMEWORK
from dragent_tests.helpers import GENERATE_PATH
from dragent_tests.helpers import collect_text
from dragent_tests.helpers import make_generate_payload


@pytest.mark.skipif(
    FRAMEWORK == "nat",
    reason="NAT returns single response in chat completions format, and we do not yet care to fix "
    "it."
)
def test_generate_single_produces_text(http_client: httpx.Client) -> None:
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

    # THEN: there are events with text
    full_text = collect_text(response_data.events)
    assert len(full_text) > 0, "Expected non-empty text response"
