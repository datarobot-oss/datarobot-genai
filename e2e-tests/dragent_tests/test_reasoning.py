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
from datarobot_genai.core.agents.verify import validate_sequence

from dragent_tests.helpers import AGENT
from dragent_tests.helpers import GENERATE_STREAM_PATH
from dragent_tests.helpers import REASONING_EVENT_TYPES
from dragent_tests.helpers import REASONING_TESTS
from dragent_tests.helpers import collect_ag_ui_events
from dragent_tests.helpers import collect_reasoning
from dragent_tests.helpers import collect_text
from dragent_tests.helpers import make_generate_payload
from dragent_tests.helpers import parse_sse_responses

# Reasoning is only emitted by the langgraph and llama_index agents, and only with a specific model.
# Skip otherwise
if not REASONING_TESTS:
    pytest.skip(
        "Reasoning tests run only against the LLM Gateway.",
        allow_module_level=True,
    )

if AGENT not in ("langgraph", "llamaindex"):
    pytest.skip(
        "Reasoning is only emitted by the langgraph and llamaindex agents.",
        allow_module_level=True,
    )


def test_generate_streaming_emits_reasoning(http_client: httpx.Client) -> None:
    """With thinking enabled, the agent streams AG-UI reasoning events alongside its answer."""
    # GIVEN: a prompt that nudges the agent to reason before answering
    payload = make_generate_payload(
        "Think it through, then outline a short blog post about why the sky is blue."
    )

    # WHEN: the payload is streamed to the generate endpoint
    with http_client.stream("POST", GENERATE_STREAM_PATH, json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        sse_responses = parse_sse_responses(response)

    ag_ui_events = collect_ag_ui_events(sse_responses)

    # THEN: the events form a valid AG-UI sequence
    validate_sequence(ag_ui_events)

    # THEN: at least one reasoning event is emitted, carrying non-empty reasoning content
    reasoning_events = [event for event in ag_ui_events if event.type in REASONING_EVENT_TYPES]
    assert reasoning_events, (
        "Expected at least one AG-UI reasoning event when thinking is enabled; "
        f"got event types {sorted({event.type for event in ag_ui_events})}"
    )
    assert collect_reasoning(ag_ui_events).strip(), "Expected non-empty reasoning content"

    # THEN: the agent still produces a normal text answer alongside its reasoning
    assert collect_text(ag_ui_events).strip(), "Expected a text response alongside the reasoning"
