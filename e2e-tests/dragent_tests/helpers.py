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
import os
import uuid

import httpx
from ag_ui.core import Event
from ag_ui.core import EventType
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse

BASE_URL = "http://localhost:8080"

GENERATE_STREAM_PATH = "/generate/stream"
GENERATE_PATH = "/generate"

AGENT = os.environ.get("AGENT")
AGENT_SUPPORTS_TOOL_CALLS = AGENT in ["langgraph", "nat", "llamaindex", "crewai"]
AGENT_SUPPORTS_TOOL_CALLS_STREAMING = AGENT in ["langgraph", "nat", "llamaindex"]

LLM = os.environ.get("LLM")

ALL_TEST_CASES = LLM == "llmgw"


def make_generate_payload(content: str) -> dict:  # type: ignore[type-arg]
    uid = uuid.uuid4().hex[:8]
    return {
        "threadId": f"test-{uid}",
        "runId": f"run-{uid}",
        "messages": [{"role": "user", "content": content, "id": f"msg-{uid}"}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
        "state": {},
    }


def parse_sse_responses(response: httpx.Response) -> list[DRAgentEventResponse]:  # type: ignore[type-arg]
    """Parse SSE text/event-stream into list of DRAgentEventResponse dicts."""
    responses = []
    for line in response.iter_lines():
        line = line.strip()  # noqa: PLW2901
        if line.startswith("data: "):
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            responses.append(DRAgentEventResponse.model_validate_json(data))
    return responses


def collect_ag_ui_events(responses: list[DRAgentEventResponse]) -> list[Event]:  # type: ignore[type-arg]
    """Flatten AG-UI events from all SSE chunks."""
    all_events: list[Event] = []
    for response in responses:
        all_events.extend(response.events)
    return all_events


def collect_text(ag_ui_events: list[Event]) -> str:  # type: ignore[type-arg]
    """Join text deltas from text message events."""
    parts = []
    for event in ag_ui_events:
        if event.type in (EventType.TEXT_MESSAGE_CONTENT, EventType.TEXT_MESSAGE_CHUNK):
            parts.append(event.delta)
    return "".join(parts)
