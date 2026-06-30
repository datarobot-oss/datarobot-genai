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

"""E2E tests for the DataRobot Memory Service through DRAgent.

Exercises the full HTTP path: ``streaming_memory_agent`` stores the user's
message via ``dr_mem0_memory`` (backed by a real ``MemorySpace``), then a
follow-up turn retrieves the fact and the inner agent recalls it in the
response.

Requires ``cases/memory.yaml`` (or equivalent env): ``WORKFLOW_FILE=workflow-memory.yaml``
and ``AGENT_MEMORY_SPACE_ID`` set before the dragent server starts
(``E2E_PROVISION_MEMORY_SPACE=true`` handles that in ``run_local.py`` / CI).
Works with any dragent agent framework (NAT, CrewAI, LangGraph, LlamaIndex).
"""

from __future__ import annotations

import os
import time
import uuid

import httpx
import pytest
from datarobot_genai.core.agents.verify import validate_sequence

from dragent_tests.helpers import WORKFLOW_FILE
from dragent_tests.helpers import collect_ag_ui_events
from dragent_tests.helpers import collect_text
from dragent_tests.helpers import make_generate_payload
from dragent_tests.helpers import stream_sse_responses

if WORKFLOW_FILE != "workflow-memory.yaml":
    pytest.skip(
        f"Memory e2e tests require WORKFLOW_FILE=workflow-memory.yaml (got {WORKFLOW_FILE!r}).",
        allow_module_level=True,
    )

if not os.environ.get("AGENT_MEMORY_SPACE_ID"):
    pytest.skip(
        "AGENT_MEMORY_SPACE_ID must be set before the dragent server starts.",
        allow_module_level=True,
    )

_RECALL_POLL_TIMEOUT_S = 90.0
_RECALL_POLL_INTERVAL_S = 3.0


def _store_secret(http_client: httpx.Client, secret: str) -> None:
    """Send a turn that asks the agent to remember *secret*."""
    store_prompt = (
        f"My e2e memory secret code is {secret}. "
        "Remember this code exactly for future conversations."
    )
    events = collect_ag_ui_events(
        stream_sse_responses(http_client, make_generate_payload(store_prompt))
    )
    validate_sequence(events)
    assert collect_text(events), "Expected a non-empty acknowledgement on the store turn."


def _poll_recall(http_client: httpx.Client, secret: str) -> str:
    """Poll the recall turn until *secret* appears in the assistant text."""
    recall_prompt = (
        "What is my e2e memory secret code? Reply with ONLY the code and nothing else."
    )
    deadline = time.monotonic() + _RECALL_POLL_TIMEOUT_S
    last_text = ""
    while time.monotonic() < deadline:
        events = collect_ag_ui_events(
            stream_sse_responses(http_client, make_generate_payload(recall_prompt))
        )
        validate_sequence(events)
        last_text = collect_text(events)
        if secret in last_text:
            return last_text
        time.sleep(_RECALL_POLL_INTERVAL_S)
    return last_text


def test_dr_memory_service_store_and_recall(http_client: httpx.Client) -> None:
    """A fact stored on one turn is recalled on a later turn via the DR Memory Service."""
    secret = f"E2E-MEM-{uuid.uuid4().hex[:10]}"

    # WHEN: the user shares a secret to remember.
    _store_secret(http_client, secret)

    # THEN: a later turn retrieves the secret from the DataRobot Memory Service.
    answer = _poll_recall(http_client, secret)
    assert secret in answer, (
        f"Expected secret {secret!r} in recall response within "
        f"{_RECALL_POLL_TIMEOUT_S}s, got: {answer!r}"
    )
