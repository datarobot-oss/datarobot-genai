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
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse

from dragent_tests.helpers import GENERATE_PATH
from dragent_tests.helpers import collect_text
from dragent_tests.helpers import make_generate_payload

if os.environ.get("AGENT") != "router":
    pytest.skip("Router e2e tests require AGENT=router", allow_module_level=True)


def test_router_single_response(http_client: httpx.Client) -> None:
    """Non-streaming /generate returns a valid non-empty AG-UI sequence."""
    # GIVEN: a user message for the router-backed agent
    payload = make_generate_payload("Say 'hello' and nothing else.")

    # WHEN: the client calls the non-streaming generate endpoint
    response = http_client.post(GENERATE_PATH, json=payload)
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")

    # THEN: the response is valid DRAgentEventResponse
    response_data = DRAgentEventResponse.model_validate_json(response.text)
    assert len(response_data.events) > 0

    # THEN: the events form a valid AG-UI sequence with non-empty text
    validate_sequence(response_data.events)
    full_text = collect_text(response_data.events)
    assert len(full_text) > 0, "Expected non-empty text response"
