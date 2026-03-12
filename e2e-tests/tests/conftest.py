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

import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Generator

import httpx
import pytest
from dotenv import load_dotenv

# Load .env from e2e-tests root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Default to a fast gateway model if not explicitly configured
if not os.environ.get("LLM_DEFAULT_MODEL"):
    os.environ["LLM_DEFAULT_MODEL"] = "azure/gpt-4o-mini"

AGENT_PORT = 8099
MCP_AGENT_PORT = 8098
GENERATE_STREAM_PATH = "/generate/stream"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--framework", default=None, help="Agent framework to test")


@pytest.fixture(scope="session")
def framework(request: pytest.FixtureRequest) -> str:
    return (
        request.config.getoption("--framework")
        or os.environ.get("DRAGENT_FRAMEWORK", "langgraph")
    )


def _start_server(
    framework: str,
    workflow_file: str,
    port: int,
    label: str,
) -> Generator[subprocess.Popen]:  # type: ignore[type-arg]
    """Start a NAT dragent server and wait for /health to respond."""
    cwd = os.path.join(os.path.dirname(__file__), "..", "dragent", framework)
    url = f"http://localhost:{port}"

    log_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        prefix=f"dragent-{label}-", suffix=".log", delete=True, mode="w"
    )

    try:
        proc = subprocess.Popen(  # noqa: S603
            [
                "uv", "run", "nat", "start", "dragent_fastapi",
                "--config_file", workflow_file,
                "--port", str(port),
            ],
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"{url}/health", timeout=2)
                if r.status_code == 200:
                    break
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(2)
        else:
            proc.kill()
            proc.wait()
            os.fsync(log_file.fileno())  # subprocess writes to fd directly; flush won't help
            log_tail = Path(log_file.name).read_text()[-2000:]
            pytest.fail(f"{label} server did not start within 90s.\nLog tail:\n{log_tail}")

        yield proc

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        log_file.close()


@pytest.fixture(scope="session")
def dragent_server(
    framework: str,
) -> Generator[subprocess.Popen]:  # type: ignore[type-arg]
    """Core dragent server (no MCP)."""
    yield from _start_server(framework, "workflow.yaml", AGENT_PORT, framework)


@pytest.fixture(scope="session")
def dragent_mcp_server(
    framework: str,
) -> Generator[subprocess.Popen]:  # type: ignore[type-arg]
    """Dragent server with MCP enabled. Only started when requested."""
    yield from _start_server(
        framework, "workflow_with_mcp.yaml", MCP_AGENT_PORT, f"{framework}-mcp"
    )


@pytest.fixture(scope="session")
def http_client(dragent_server: subprocess.Popen) -> Generator[httpx.Client]:  # type: ignore[type-arg]
    timeout = httpx.Timeout(connect=10, read=300, write=10, pool=10)
    with httpx.Client(base_url=f"http://localhost:{AGENT_PORT}", timeout=timeout) as client:
        yield client


@pytest.fixture(scope="session")
def mcp_http_client(dragent_mcp_server: subprocess.Popen) -> Generator[httpx.Client]:  # type: ignore[type-arg]
    timeout = httpx.Timeout(connect=10, read=300, write=10, pool=10)
    with httpx.Client(base_url=f"http://localhost:{MCP_AGENT_PORT}", timeout=timeout) as client:
        yield client


# --- Helpers ---


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


def parse_sse_events(response: httpx.Response) -> list[dict]:  # type: ignore[type-arg]
    """Parse SSE text/event-stream into list of DRAgentEventResponse dicts."""
    events = []
    for line in response.iter_lines():
        line = line.strip()
        if line.startswith("data: "):
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            try:
                events.append(json.loads(data))
            except json.JSONDecodeError:
                continue
    return events


def collect_ag_ui_events(sse_events: list[dict]) -> list[dict]:  # type: ignore[type-arg]
    """Flatten AG-UI events from all SSE chunks."""
    all_events: list[dict] = []  # type: ignore[type-arg]
    for sse in sse_events:
        all_events.extend(sse.get("events", []))
    return all_events


def collect_text(ag_ui_events: list[dict]) -> str:  # type: ignore[type-arg]
    """Join text deltas from text message events."""
    parts = []
    for event in ag_ui_events:
        if event.get("type") in ("TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_CHUNK"):
            parts.append(event.get("delta", ""))
    return "".join(parts)
