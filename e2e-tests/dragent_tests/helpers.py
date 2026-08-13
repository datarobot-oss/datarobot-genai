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
from typing import Any

import httpx
import litellm
import yaml
from ag_ui.core import Event
from ag_ui.core import EventType
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse

BASE_URL = "http://localhost:8080"

GENERATE_STREAM_PATH = "/generate/stream"
GENERATE_PATH = "/generate"

AGENT = os.environ.get("AGENT", "base")
AGENT_SUPPORTS_TOOL_CALLS = AGENT in ["langgraph", "nat", "llamaindex", "crewai"]
AGENT_SUPPORTS_TOOL_CALLS_STREAMING = AGENT in ["langgraph", "nat", "llamaindex", "crewai"]

LLM = os.environ.get("LLM", "llmgw")
LLM_DEFAULT_MODEL = os.environ.get("LLM_DEFAULT_MODEL")

WORKFLOW_FILE = os.environ.get("WORKFLOW_FILE", "workflow.yaml")

E2E_ROOT = Path(__file__).resolve().parent.parent
RUNNER_MODULE = "dragent.run_agent"


def llm_supports_reasoning(llm_default_model: str) -> bool:
    llm_default_model = llm_default_model.removeprefix("datarobot/")
    return litellm.supports_reasoning(llm_default_model)


def _merge_workflow_llm_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    llms = dict(data.get("llms") or {})
    base_name = data.get("base")
    if base_name:
        base_data = yaml.safe_load((path.parent / base_name).read_text()) or {}
        base_llms = dict(base_data.get("llms") or {})
        for name, overlay in llms.items():
            merged = dict(base_llms.get(name) or {})
            merged.update(overlay)
            llms[name] = merged
    return llms.get("datarobot_llm") or {}


def workflow_reasoning_enabled() -> bool:
    llm_cfg = _merge_workflow_llm_config(workflow_file())
    if llm_cfg.get("reasoning") is True:
        return True

    extra_body = llm_cfg.get("extra_body") or {}
    thinking = extra_body.get("thinking") or {}
    if thinking.get("type") == "enabled":
        return True
    if extra_body.get("thinking_config"):
        return True

    reasoning_effort = extra_body.get("reasoning_effort")
    return bool(reasoning_effort and reasoning_effort != "none")


def should_run_reasoning_test() -> bool:
    return (
        LLM_DEFAULT_MODEL
        and llm_supports_reasoning(LLM_DEFAULT_MODEL)
        and AGENT in ("langgraph", "llamaindex")
        and workflow_reasoning_enabled()
    )


def agent_dir() -> Path:
    """Path to the dragent agent config directory for *agent* (defaults to AGENT/base)."""
    return E2E_ROOT / "dragent" / AGENT


def workflow_file() -> Path:
    """Path to the dragent agent workflow file for *agent* (defaults to WORKFLOW_FILE)."""
    return agent_dir() / WORKFLOW_FILE


def build_chat_completion(
    content: str = "Say 'hello world' and nothing else.", stream: bool = False
) -> dict:  # type: ignore[type-arg]
    """Build the one-shot chat completion payload used by the inline runner tests."""
    return {
        "model": "unknown",
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
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


def raise_if_nat_workflow_error_payload(payload: str) -> None:
    """Fail if *payload* is a bare NAT FastAPI workflow-error JSON object.

    NAT's streaming helpers catch workflow exceptions and yield
    ``Error(...).model_dump_json()`` **without** an SSE ``data:`` prefix
    (see ``nat.front_ends.fastapi.response_helpers``).
    """
    text = payload.strip()
    if not text or not text.startswith("{"):
        return
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(body, dict):
        return
    code = body.get("code")
    if code != "workflow_error":
        return
    message = body.get("message") or text
    details = body.get("details") or ""
    raise AssertionError(
        "NAT streaming endpoint reported a workflow_error (not an SSE data "
        f"frame). details={details!r} message={message!r}. "
    )


def parse_sse_responses(response: httpx.Response) -> list[DRAgentEventResponse]:  # type: ignore[type-arg]
    """Parse SSE text/event-stream into list of DRAgentEventResponse dicts.

    Raises ``AssertionError`` if the stream includes a bare NAT
    ``workflow_error`` JSON payload (no ``data:`` prefix).
    """
    responses = []
    for line in response.iter_lines():
        line = line.strip()  # noqa: PLW2901
        if not line:
            continue
        if line.startswith("data: "):
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            # Some error paths may incorrectly prefix the NAT Error object.
            raise_if_nat_workflow_error_payload(data)
            responses.append(DRAgentEventResponse.model_validate_json(data))
            continue
        # Bare JSON line (no SSE framing) — fail loudly on workflow errors.
        raise_if_nat_workflow_error_payload(line)
    return responses


def stream_sse_responses(http_client: httpx.Client, payload: dict) -> list[DRAgentEventResponse]:  # type: ignore[type-arg]
    """POST ``payload`` to the streaming generate endpoint and parse the SSE stream.

    Asserts a 200 ``text/event-stream`` response, then returns the parsed
    DRAgentEventResponse chunks. Shared by the streaming e2e tests.

    Also fails if NAT emits a bare ``workflow_error`` JSON line (see
    ``raise_if_nat_workflow_error_payload``).
    """
    with http_client.stream("POST", GENERATE_STREAM_PATH, json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        return parse_sse_responses(response)


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
