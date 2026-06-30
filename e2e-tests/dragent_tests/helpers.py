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
import litellm
from ag_ui.core import Event
from ag_ui.core import EventType
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse

from dragent_tests.mock_otel_collector import OTLP_TRACES_PATH
from dragent_tests.mock_otel_collector import MockOtelCollector

BASE_URL = "http://localhost:8080"

GENERATE_STREAM_PATH = "/generate/stream"
GENERATE_PATH = "/generate"

# --- OpenTelemetry tracing -------------------------------------------------
# The dragent server (started before pytest) is configured to export spans to
# the in-process mock collector via OTEL_EXPORTER_OTLP_ENDPOINT /
# OTEL_EXPORTER_OTLP_HEADERS. These constants MUST match the values the server
# is launched with in ``e2e-tests/dragent/Taskfile.yaml``.
MOCK_OTEL_COLLECTOR_PORT = int(os.environ.get("MOCK_OTEL_COLLECTOR_PORT", "4318"))
# OTLP base endpoint the server exports to; ``/v1/traces`` is appended by the
# DataRobot exporter (see resolve_otel_traces_endpoint_from_env).
OTEL_EXPORTER_OTLP_ENDPOINT = f"http://localhost:{MOCK_OTEL_COLLECTOR_PORT}/otel"
# Sentinel DataRobot auth headers; the mock collector does not validate them,
# the tests only assert they reached the ingest unmodified.
OTEL_API_KEY = "e2e-otel-token"
OTEL_ENTITY_ID = "deployment-e2e-test"
OTEL_EXPORTER_OTLP_HEADERS = (
    f"X-DataRobot-Api-Key={OTEL_API_KEY},X-DataRobot-Entity-Id={OTEL_ENTITY_ID}"
)

# Span attributes that map to the deployment Tracing table columns, per
# https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-tracing-code.html#map-spans-and-attributes-to-the-tracing-table
# Mirrors the constants in
# ``datarobot_genai.dragent.plugins.datarobot_otel_conventions_middleware``.
GEN_AI_PROMPT = "gen_ai.prompt"  # Prompt column
GEN_AI_COMPLETION = "gen_ai.completion"  # Completion column
TOOL_NAME = "tool_name"  # Tools column

AGENT = os.environ.get("AGENT", "base")
AGENT_SUPPORTS_TOOL_CALLS = AGENT in ["langgraph", "nat", "llamaindex", "crewai"]
AGENT_SUPPORTS_TOOL_CALLS_STREAMING = AGENT in ["langgraph", "nat", "llamaindex"]

LLM = os.environ.get("LLM", "llmgw")
LLM_DEFAULT_MODEL = os.environ.get("LLM_DEFAULT_MODEL")

WORKFLOW_FILE = os.environ.get("WORKFLOW_FILE", "workflow.yaml")

E2E_ROOT = Path(__file__).resolve().parent.parent
RUNNER_MODULE = "dragent.run_agent"


def llm_supports_reasoning(llm_default_model: str) -> bool:
    llm_default_model = llm_default_model.removeprefix("datarobot/")
    return litellm.supports_reasoning(llm_default_model)


def should_run_reasoning_test() -> bool:
    return (
        LLM_DEFAULT_MODEL
        and llm_supports_reasoning(LLM_DEFAULT_MODEL)
        and AGENT in ("langgraph", "llamaindex")
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


def stream_sse_responses(http_client: httpx.Client, payload: dict) -> list[DRAgentEventResponse]:  # type: ignore[type-arg]
    """POST ``payload`` to the streaming generate endpoint and parse the SSE stream.

    Asserts a 200 ``text/event-stream`` response, then returns the parsed
    DRAgentEventResponse chunks. Shared by the streaming e2e tests.
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


def _assert_datarobot_auth_headers(collector: MockOtelCollector) -> None:
    """At least one trace export must carry the DataRobot ingest auth headers.

    HTTP header names are case-insensitive, so compare on lowercased keys
    rather than relying on the exporter's exact casing.
    """

    def lower(headers: dict[str, str]) -> dict[str, str]:
        return {k.lower(): v for k, v in headers.items()}

    authed = [
        request
        for request in collector.requests
        if request.path == OTLP_TRACES_PATH
        and lower(request.headers).get("x-datarobot-api-key") == OTEL_API_KEY
        and lower(request.headers).get("x-datarobot-entity-id") == OTEL_ENTITY_ID
    ]
    assert authed, (
        f"Expected ≥ 1 POST to {OTLP_TRACES_PATH} with DR auth headers "
        f"(X-DataRobot-Api-Key={OTEL_API_KEY}, X-DataRobot-Entity-Id={OTEL_ENTITY_ID}); "
        f"captured {len(collector.requests)} request(s) at paths "
        f"{sorted({r.path for r in collector.requests})} with header keys "
        f"{[sorted(r.headers.keys()) for r in collector.requests if r.path == OTLP_TRACES_PATH]}."
    )


def assert_tracing_conventions(
    collector: MockOtelCollector,
    prompt: str,
    *,
    expect_tool_name: bool = False,
    timeout: float = 30.0,
) -> None:
    """Assert the agent exported DataRobot Tracing-table spans for *prompt*.

    Finds this request's spans by ``gen_ai.prompt`` (the verbatim user message
    recorded by the ``datarobot_otel_conventions`` middleware), then asserts the
    convention attributes are present:

    * ``gen_ai.prompt`` and ``gen_ai.completion`` on the per-invocation
      ``datarobot_agent`` span, and
    * the export carried the DataRobot ingest auth headers.

    When *expect_tool_name* is set, also requires a ``tool_name`` span in the
    same trace (frameworks that surface tool calls as AG-UI events).
    """
    agent_spans = collector.wait_for_spans(
        lambda span: span.attributes.get(GEN_AI_PROMPT) == prompt
        and GEN_AI_COMPLETION in span.attributes,
        timeout=timeout,
    )
    if not agent_spans:
        observed = sorted(
            str(s.attributes[GEN_AI_PROMPT])
            for s in collector.spans()
            if GEN_AI_PROMPT in s.attributes
        )
        raise AssertionError(
            f"No exported span carried {GEN_AI_PROMPT}=={prompt!r} together with "
            f"{GEN_AI_COMPLETION}. Observed prompts: {observed}"
        )

    _assert_datarobot_auth_headers(collector)

    if expect_tool_name:
        trace_ids = {span.trace_id for span in agent_spans}
        spans_in_trace = [s for s in collector.spans() if s.trace_id in trace_ids]
        tool_spans = [s for s in spans_in_trace if s.attributes.get(TOOL_NAME)]
        assert tool_spans, (
            f"Expected a span carrying {TOOL_NAME!r} in the same trace as the "
            f"agent span for the tool-calling request; "
            f"got span names {sorted({s.name for s in spans_in_trace})}."
        )
