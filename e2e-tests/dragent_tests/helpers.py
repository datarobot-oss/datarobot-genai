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
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

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

ALL_TEST_CASES = os.environ.get("ALL_TEST_CASES") == "true"

E2E_ROOT = Path(__file__).resolve().parent.parent
RUNNER_MODULE = "dragent.run_agent"


def agent_dir(agent: str | None = None) -> Path:
    """Path to the dragent agent config directory for *agent* (defaults to AGENT/base)."""
    return E2E_ROOT / "dragent" / (agent or AGENT or "base")


def build_chat_completion(content: str = "Say 'hello world' and nothing else.") -> dict:  # type: ignore[type-arg]
    """Build the one-shot chat completion payload used by the inline runner tests."""
    return {
        "model": "unknown",
        "messages": [{"role": "user", "content": content}],
    }


def spawn_runner(
    *,
    chat_completion: dict[str, object],
    output_path: Path,
    custom_model_dir: Path | None = None,
    config_file: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 180.0,
) -> subprocess.CompletedProcess[str]:
    """Spawn ``python -m dragent.run_agent`` and return the completed process.

    Invoked as ``-m`` (not as a script) so ``cwd=e2e-tests`` lands on
    ``sys.path`` instead of ``e2e-tests/dragent/``; the latter contains a local
    ``nat/`` subpackage that otherwise shadows the third-party ``nvidia-nat``
    and breaks ``import nat.data_models``.
    """
    custom_model_dir = custom_model_dir or agent_dir()
    cmd = [
        sys.executable,
        "-m",
        RUNNER_MODULE,
        "--chat_completion",
        json.dumps(chat_completion),
        "--custom_model_dir",
        str(custom_model_dir),
        "--output_path",
        str(output_path),
    ]
    if config_file is not None:
        cmd.extend(["--config_file", str(config_file)])
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(E2E_ROOT),
        env=env if env is not None else {**os.environ},
        check=False,
    )


# Prompt/response OOTB token_count guards only in e2e dragent workflow moderation blocks.
EXPECTED_DATAROBOT_MODERATION_TOKEN_KEYS = frozenset(
    {
        "Prompts_token_count",
        "Responses_token_count",
    }
)


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
