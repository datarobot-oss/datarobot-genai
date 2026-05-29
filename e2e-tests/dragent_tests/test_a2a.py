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

"""End-to-end tests for the Agent-to-Agent (A2A) protocol.

These tests validate that the A2A server is reachable, advertises a valid
agent card, and can process a ``message/send`` JSON-RPC request end-to-end.

A2A is enabled in every agent's default ``workflow.yaml``, so these tests
run for all agent types (NAT, LangGraph, CrewAI, LlamaIndex, base).
"""

from __future__ import annotations

import httpx

from dragent_tests.helpers import A2A_AGENT_CARD_PATH
from dragent_tests.helpers import A2A_PATH
from dragent_tests.helpers import make_a2a_message_send_payload


def test_a2a_agent_card(http_client: httpx.Client) -> None:
    """The agent card endpoint returns a well-formed card with required fields."""
    response = http_client.get(A2A_AGENT_CARD_PATH)

    assert response.status_code == 200, (
        f"Expected 200 from agent card endpoint, got {response.status_code}: "
        f"{response.text[:500]}"
    )

    card = response.json()
    assert card.get("name"), "Agent card missing 'name'"
    assert card.get("url"), "Agent card missing 'url'"
    assert card.get("skills"), "Agent card missing 'skills' or skills list is empty"


def test_a2a_message_send(http_client: httpx.Client) -> None:
    """A message/send request is processed and returns a JSON-RPC result."""
    payload = make_a2a_message_send_payload("Say 'hello world' and nothing else.")

    response = http_client.post(A2A_PATH, json=payload)

    assert response.status_code == 200, (
        f"Expected 200 from A2A endpoint, got {response.status_code}: "
        f"{response.text[:500]}"
    )

    data = response.json()

    # Valid JSON-RPC 2.0 envelope
    assert data.get("jsonrpc") == "2.0", f"Expected jsonrpc 2.0, got {data.get('jsonrpc')}"
    assert data.get("id") == payload["id"], (
        f"JSON-RPC id mismatch: expected {payload['id']}, got {data.get('id')}"
    )

    # No error
    assert "error" not in data, f"JSON-RPC error: {data.get('error')}"

    # Result present with a completed status
    result = data.get("result")
    assert result is not None, f"JSON-RPC response missing 'result'. Keys: {sorted(data.keys())}"
    assert result.get("status", {}).get("state") == "completed", (
        f"Expected task state 'completed', got: {result.get('status')}"
    )


