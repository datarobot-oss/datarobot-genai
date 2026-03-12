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

from tests.conftest import GENERATE_STREAM_PATH
from tests.conftest import collect_ag_ui_events
from tests.conftest import collect_text
from tests.conftest import make_generate_payload
from tests.conftest import parse_sse_events


# --- Health checks ---


def test_health_endpoint(http_client: httpx.Client) -> None:
    r = http_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_ping_endpoint(http_client: httpx.Client) -> None:
    r = http_client.get("/ping")
    assert r.status_code == 200


# --- Streaming (single LLM call shared across assertions) ---


@pytest.fixture(scope="module")
def streaming_response(http_client: httpx.Client) -> dict:  # type: ignore[type-arg]
    """Single streaming call shared by all streaming tests."""
    payload = make_generate_payload("Say 'hello world' and nothing else.")
    with http_client.stream("POST", GENERATE_STREAM_PATH, json=payload) as response:
        status = response.status_code
        content_type = response.headers.get("content-type", "")
        sse_events = parse_sse_events(response)
    return {"status_code": status, "content_type": content_type, "sse_events": sse_events}


@pytest.fixture(scope="module")
def streaming_ag_ui_events(streaming_response: dict) -> list[dict]:  # type: ignore[type-arg]
    return collect_ag_ui_events(streaming_response["sse_events"])


def test_generate_streaming_returns_sse(streaming_response: dict) -> None:  # type: ignore[type-arg]
    """POST /generate/stream returns SSE content type."""
    assert streaming_response["status_code"] == 200
    assert "text/event-stream" in streaming_response["content_type"]
    assert len(streaming_response["sse_events"]) > 0


def test_generate_streaming_has_ag_ui_events(streaming_ag_ui_events: list[dict]) -> None:  # type: ignore[type-arg]
    """SSE chunks contain AG-UI events with a 'type' field."""
    assert len(streaming_ag_ui_events) > 0
    assert all("type" in e for e in streaming_ag_ui_events)


def test_generate_streaming_produces_text(streaming_ag_ui_events: list[dict]) -> None:  # type: ignore[type-arg]
    """Concatenated text deltas produce a non-empty response."""
    full_text = collect_text(streaming_ag_ui_events)
    assert len(full_text) > 0, "Expected non-empty text response"


def test_generate_streaming_has_text_message_events(streaming_ag_ui_events: list[dict]) -> None:  # type: ignore[type-arg]
    """Stream contains TEXT_MESSAGE_* event types."""
    event_types = {e["type"] for e in streaming_ag_ui_events}
    text_types = {
        "TEXT_MESSAGE_START", "TEXT_MESSAGE_CHUNK",
        "TEXT_MESSAGE_END", "TEXT_MESSAGE_CONTENT",
    }
    assert event_types & text_types, f"No text message events found. Got: {event_types}"
