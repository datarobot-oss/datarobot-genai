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


@pytest.fixture(scope="session")
def streaming_ag_ui_events(http_client: httpx.Client) -> list[dict]:  # type: ignore[type-arg]
    """Single streaming call shared by all streaming tests."""
    payload = make_generate_payload("Say 'hello world' and nothing else.")
    with http_client.stream("POST", GENERATE_STREAM_PATH, json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        sse_events = parse_sse_events(response)
    return collect_ag_ui_events(sse_events)


def test_generate_streaming_produces_text(streaming_ag_ui_events: list[dict]) -> None:  # type: ignore[type-arg]
    """Concatenated text deltas produce a non-empty response."""
    full_text = collect_text(streaming_ag_ui_events)
    assert len(full_text) > 0, "Expected non-empty text response"
