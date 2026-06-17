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
from ag_ui.core import Event
from ag_ui.core import EventType
from datarobot_genai.core.agents.verify import validate_sequence

from dragent_tests.helpers import AGENT
from dragent_tests.helpers import collect_ag_ui_events
from dragent_tests.helpers import collect_text
from dragent_tests.helpers import make_generate_payload
from dragent_tests.helpers import stream_sse_responses

if AGENT not in ("langgraph", "llamaindex"):
    pytest.skip(
        "Reasoning is only emitted by the langgraph and llamaindex agents.",
        allow_module_level=True,
    )

# AG-UI event types that carry a reasoning/thinking step from a reasoning-capable model.
REASONING_EVENT_TYPES = frozenset(
    {
        EventType.REASONING_MESSAGE_START,
        EventType.REASONING_MESSAGE_CONTENT,
        EventType.REASONING_MESSAGE_CHUNK,
        EventType.REASONING_MESSAGE_END,
    }
)


def collect_reasoning(ag_ui_events: list[Event]) -> str:  # type: ignore[type-arg]
    """Join reasoning/thinking deltas from AG-UI reasoning events.

    langgraph and llama_index both emit incremental reasoning as
    ``REASONING_MESSAGE_CHUNK`` events; ``*_CONTENT`` variants are included so
    the helper stays correct if an agent emits the start/content/end form.
    """
    parts = []
    for event in ag_ui_events:
        if event.type in (
            EventType.REASONING_MESSAGE_CONTENT,
            EventType.REASONING_MESSAGE_CHUNK,
            EventType.THINKING_TEXT_MESSAGE_CONTENT,
        ):
            delta = getattr(event, "delta", None)
            if delta:
                parts.append(delta)
    return "".join(parts)


def test_generate_streaming_emits_reasoning(http_client: httpx.Client) -> None:
    """With thinking enabled, the agent streams AG-UI reasoning events alongside its answer."""
    # GIVEN: a prompt that nudges the agent to reason before answering
    payload = make_generate_payload(
        "Think it through, then outline a short blog post about why the sky is blue."
    )

    # WHEN: the payload is streamed to the generate endpoint
    sse_responses = stream_sse_responses(http_client, payload)

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
