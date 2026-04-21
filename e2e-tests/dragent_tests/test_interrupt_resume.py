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

import uuid

import httpx
import pytest
from ag_ui.core import EventType

from dragent.langgraph.myagent import (
    E2E_INTERRUPT_CANCELLED,
    E2E_INTERRUPT_CONTINUING,
)
from dragent_tests.helpers import AGENT
from dragent_tests.helpers import ALL_TEST_CASES
from dragent_tests.helpers import GENERATE_STREAM_PATH
from dragent_tests.helpers import collect_ag_ui_events
from dragent_tests.helpers import collect_text
from dragent_tests.helpers import parse_sse_responses

pytestmark = pytest.mark.skipif(
    AGENT != "langgraph",
    reason="Interrupt/resume E2E uses dragent/langgraph (set AGENT=langgraph).",
)

if not ALL_TEST_CASES:
    pytest.skip(
        "Running minimal test set for non-LLM Gateway LLM, skipping interrupt E2E",
        allow_module_level=True,
    )


def test_stream_interrupt_then_resume_plain_user_message_no(http_client: httpx.Client) -> None:
    """First request pauses on interrupt; second sends only 'no' and gets the cancel branch."""
    uid = uuid.uuid4().hex[:8]
    thread_id = f"e2e-hitl-{uid}"

    payload_interrupt = {
        "threadId": thread_id,
        "runId": f"run-a-{uid}",
        "messages": [
            {"role": "user", "content": '{"topic": "start"}', "id": f"m1-{uid}"},
        ],
        "tools": [],
        "context": [],
        "forwardedProps": {},
        "state": {},
    }
    with http_client.stream("POST", GENERATE_STREAM_PATH, json=payload_interrupt) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        sse_1 = parse_sse_responses(response)

    events_1 = collect_ag_ui_events(sse_1)
    custom_names = [getattr(e, "name", None) for e in events_1 if e.type == EventType.CUSTOM]
    assert "langgraph.interrupt" in custom_names

    finished_1 = [e for e in events_1 if e.type == EventType.RUN_FINISHED][-1]
    assert finished_1.result is not None
    assert finished_1.result["langgraph"]["interrupted"] is True

    payload_resume = {
        "threadId": thread_id,
        "runId": f"run-b-{uid}",
        "messages": [
            {"role": "user", "content": "no", "id": f"m2-{uid}"},
        ],
        "tools": [],
        "context": [],
        "forwardedProps": {},
        "state": {},
    }
    with http_client.stream("POST", GENERATE_STREAM_PATH, json=payload_resume) as response:
        assert response.status_code == 200
        sse_2 = parse_sse_responses(response)

    events_2 = collect_ag_ui_events(sse_2)
    text = collect_text(events_2)
    assert E2E_INTERRUPT_CANCELLED in text


def test_stream_interrupt_then_resume_plain_user_message_yes(http_client: httpx.Client) -> None:
    """First request pauses on interrupt; second sends 'yes' and continues to the writer."""
    uid = uuid.uuid4().hex[:8]
    thread_id = f"e2e-hitl-yes-{uid}"

    payload_interrupt = {
        "threadId": thread_id,
        "runId": f"run-a-{uid}",
        "messages": [
            {"role": "user", "content": '{"topic": "start"}', "id": f"m1-{uid}"},
        ],
        "tools": [],
        "context": [],
        "forwardedProps": {},
        "state": {},
    }
    with http_client.stream("POST", GENERATE_STREAM_PATH, json=payload_interrupt) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        sse_1 = parse_sse_responses(response)

    events_1 = collect_ag_ui_events(sse_1)
    custom_names = [getattr(e, "name", None) for e in events_1 if e.type == EventType.CUSTOM]
    assert "langgraph.interrupt" in custom_names

    finished_1 = [e for e in events_1 if e.type == EventType.RUN_FINISHED][-1]
    assert finished_1.result is not None
    assert finished_1.result["langgraph"]["interrupted"] is True

    payload_resume = {
        "threadId": thread_id,
        "runId": f"run-b-{uid}",
        "messages": [
            {"role": "user", "content": "yes", "id": f"m2-{uid}"},
        ],
        "tools": [],
        "context": [],
        "forwardedProps": {},
        "state": {},
    }
    with http_client.stream("POST", GENERATE_STREAM_PATH, json=payload_resume) as response:
        assert response.status_code == 200
        sse_2 = parse_sse_responses(response)

    events_2 = collect_ag_ui_events(sse_2)
    text = collect_text(events_2)
    assert E2E_INTERRUPT_CONTINUING in text
    # Resume completed the graph (writer ran); no second interrupt on this run.
    custom_names_2 = [getattr(e, "name", None) for e in events_2 if e.type == EventType.CUSTOM]
    assert "langgraph.interrupt" not in custom_names_2
