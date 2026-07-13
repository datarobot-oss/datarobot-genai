# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import json

import httpx
import pytest
import respx

from datarobot_genai.drtools.core.sandbox import DataRobotWorkloadSandbox
from datarobot_genai.drtools.core.sandbox import SandboxError
from datarobot_genai.drtools.core.sandbox import SandboxSecurityContext
from datarobot_genai.drtools.core.sandbox import SandboxTimeout

API_BASE = "https://app.datarobot.com/api/v2"
WORKLOAD_ID = "wkl_123"
CREATE_URL = f"{API_BASE}/workloads/"
GET_URL = f"{API_BASE}/workloads/{WORKLOAD_ID}"
DELETE_URL = f"{API_BASE}/workloads/{WORKLOAD_ID}"
LOGS_URL = f"{API_BASE}/otel/workload/{WORKLOAD_ID}/logs/"


def _sandbox(client: httpx.AsyncClient | None = None) -> DataRobotWorkloadSandbox:
    return DataRobotWorkloadSandbox(
        image="datarobotdev/datarobot-user-models:public_dropin_environments_dr_mcp_execute_sandbox_minimal_latest",
        datarobot_endpoint=API_BASE,
        datarobot_api_token="test-token",
        http_client=client,
    )


def _create_response(status: str = "provisioning") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "proton_abc",
            "workloadId": WORKLOAD_ID,
            "status": status,
        },
    )


def _logs_response(message: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "count": 1,
            "next": None,
            "previous": None,
            "data": [
                {
                    "timestamp": "2026-05-13T00:00:00Z",
                    "level": "INFO",
                    "message": message,
                    "spanId": "s",
                    "traceId": "t",
                }
            ],
        },
    )


def _logs_response_entries(entries: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "count": len(entries),
            "next": None,
            "previous": None,
            "data": entries,
        },
    )


@respx.mock
async def test_happy_path_returns_value_and_strips_marker() -> None:
    submit = respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        side_effect=[
            httpx.Response(200, json={"id": WORKLOAD_ID, "status": "running"}),
            httpx.Response(
                200,
                json={
                    "id": WORKLOAD_ID,
                    "status": "succeeded",
                    "statusDetails": {"logTail": []},
                },
            ),
        ]
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response("hello\n__DR_SANDBOX_RESULT__:42"))
    delete_route = respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("_return = 42")

    assert result.return_value == 42
    assert result.stdout == "hello"
    assert submit.called
    assert delete_route.called


@respx.mock
async def test_submit_contains_security_context_camel_case_when_provided() -> None:
    captured: dict[str, object] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return _create_response("provisioning")

    respx.post(CREATE_URL).mock(side_effect=_capture)
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "succeeded"})
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response(""))
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    sb = DataRobotWorkloadSandbox(
        image="datarobotdev/datarobot-user-models:public_dropin_environments_dr_mcp_execute_sandbox_minimal_latest",
        datarobot_endpoint=API_BASE,
        datarobot_api_token="test-token",
        security_context=SandboxSecurityContext(),
    )
    async with httpx.AsyncClient() as client:
        sb._http_client = client
        await sb.run("_return = 1", inputs={"x": [1, 2]}, timeout_s=10.0)

    body = captured["body"]
    assert captured["auth"] == "Bearer test-token"
    container = body["artifact"]["spec"]["containerGroups"][0]["containers"][0]
    sc = container["securityContext"]
    assert sc["readOnlyRootFilesystem"] is True
    assert sc["allowPrivilegeEscalation"] is False
    assert sc["capabilities"] == {"drop": ["ALL"]}
    assert sc["seccompProfile"] == {"type": "RuntimeDefault"}

    env = {e["name"]: e["value"] for e in container["environmentVars"]}
    assert "DR_SANDBOX_CODE_B64" in env
    assert "DR_SANDBOX_INPUTS_B64" in env
    decoded_inputs = json.loads(base64.b64decode(env["DR_SANDBOX_INPUTS_B64"]))
    assert decoded_inputs == {"x": [1, 2]}


@respx.mock
async def test_submit_omits_security_context_when_none() -> None:
    captured: dict[str, object] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _create_response("provisioning")

    respx.post(CREATE_URL).mock(side_effect=_capture)
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "succeeded"})
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response(""))
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        await _sandbox(client).run("_return = 1")

    container = captured["body"]["artifact"]["spec"]["containerGroups"][0]["containers"][0]
    assert "securityContext" not in container


@respx.mock
async def test_workload_failure_raises_sandbox_error() -> None:
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(
            200, json={"id": WORKLOAD_ID, "status": "failed", "exitCode": 1}
        )
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response("boom"))
    delete_route = respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        with pytest.raises(SandboxError):
            await _sandbox(client).run("_return = 1")

    assert delete_route.called


@respx.mock
async def test_run_deletes_workload_on_success() -> None:
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "succeeded"})
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response("__DR_SANDBOX_RESULT__:1"))
    delete_route = respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        await _sandbox(client).run("_return = 1")

    assert delete_route.called


@respx.mock
async def test_run_deletes_workload_on_timeout() -> None:
    respx.post(CREATE_URL).mock(return_value=_create_response())
    # Always return "running" — never terminal, so polling exceeds the deadline.
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "running"})
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response(""))
    delete_route = respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        with pytest.raises(SandboxTimeout):
            await _sandbox(client).run("_return = 1", timeout_s=0.05)

    assert delete_route.called


@respx.mock
async def test_run_deletes_workload_on_cancellation() -> None:
    respx.post(CREATE_URL).mock(return_value=_create_response())

    poll_started = asyncio.Event()

    async def _slow_poll(request: httpx.Request) -> httpx.Response:
        poll_started.set()
        await asyncio.sleep(5.0)
        return httpx.Response(200, json={"id": WORKLOAD_ID, "status": "running"})

    respx.get(GET_URL).mock(side_effect=_slow_poll)
    respx.get(LOGS_URL).mock(return_value=_logs_response(""))
    delete_route = respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        task = asyncio.create_task(_sandbox(client).run("_return = 1", timeout_s=30.0))
        await poll_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert delete_route.called


@respx.mock
async def test_runner_timeout_exit_code_raises_sandbox_timeout() -> None:
    """Runner self-killed via SIGALRM cap (exit 124) surfaces as SandboxTimeout."""
    respx.post(CREATE_URL).mock(return_value=_create_response())
    # Workload-api says "succeeded" terminally — but the container exited
    # 124 because runner.py's in-process SIGALRM cap fired first. Caller
    # should still see this as a timeout, not a successful run.
    respx.get(GET_URL).mock(
        return_value=httpx.Response(
            200, json={"id": WORKLOAD_ID, "status": "succeeded", "exitCode": 124}
        )
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response("__DR_SANDBOX_RESULT__:null"))
    delete_route = respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        with pytest.raises(SandboxTimeout, match="in-process timeout"):
            await _sandbox(client).run("_return = 1", timeout_s=2.0)

    assert delete_route.called


@respx.mock
async def test_submit_passes_runner_timeout_env_var() -> None:
    """timeout_s flows through to DR_SANDBOX_TIMEOUT_SECS on the container."""
    captured: dict[str, object] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _create_response("provisioning")

    respx.post(CREATE_URL).mock(side_effect=_capture)
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "succeeded"})
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response(""))
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        await _sandbox(client).run("_return = 1", timeout_s=42.0)

    container = captured["body"]["artifact"]["spec"]["containerGroups"][0]["containers"][0]
    env = {e["name"]: e["value"] for e in container["environmentVars"]}
    assert env["DR_SANDBOX_TIMEOUT_SECS"] == "42"


@respx.mock
async def test_status_terminal_timeout_raises_sandbox_timeout() -> None:
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "timeout"})
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response(""))
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        with pytest.raises(SandboxTimeout):
            await _sandbox(client).run("_return = 1")


async def test_externals_not_supported() -> None:
    sb = DataRobotWorkloadSandbox(
        image="dr-sandbox",
        datarobot_endpoint=API_BASE,
        datarobot_api_token="t",
    )
    with pytest.raises(NotImplementedError):
        await sb.run("_return = 1", externals={"f": lambda: None})


@respx.mock
async def test_stderr_captured_from_error_level_entries() -> None:
    """ERROR-level OTEL entries surface on stderr; stdout keeps the marker line."""
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(
            200, json={"id": WORKLOAD_ID, "status": "failed", "exitCode": 1}
        )
    )
    respx.get(LOGS_URL).mock(
        return_value=_logs_response_entries(
            [
                {"level": "INFO", "message": "starting\n"},
                {"level": "ERROR", "message": "RuntimeError: boom\n"},
            ]
        )
    )
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        with pytest.raises(SandboxError) as excinfo:
            await _sandbox(client).run("raise RuntimeError('boom')")

    # The workload failed, so run() raises — but we still want to make sure the
    # log partitioning routed the error text to stderr. Exercise it directly.
    stdout, stderr = DataRobotWorkloadSandbox._partition_log_entries(
        [
            {"level": "INFO", "message": "starting\n"},
            {"level": "ERROR", "message": "RuntimeError: boom\n"},
        ]
    )
    assert stdout == "starting"
    assert "RuntimeError: boom" in stderr
    assert "failed" in str(excinfo.value)


@respx.mock
async def test_stderr_captured_and_marker_parsed_from_stdout() -> None:
    """On success, stderr is populated and the marker is still parsed off stdout."""
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "succeeded"})
    )
    respx.get(LOGS_URL).mock(
        return_value=_logs_response_entries(
            [
                {"level": "WARNING", "message": "deprecation notice\n"},
                {"level": "ERROR", "message": "some diagnostic\n"},
                {"level": "INFO", "message": "hello\n__DR_SANDBOX_RESULT__:7"},
            ]
        )
    )
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("_return = 7")

    assert result.return_value == 7
    # INFO + WARNING stay on stdout (only ERROR/CRITICAL/FATAL route to stderr);
    # the marker line is stripped from stdout by parse_result_marker.
    assert result.stdout == "deprecation notice\nhello"
    assert result.stderr == "some diagnostic"


def test_partition_log_entries_captures_stacktrace() -> None:
    """A ``stacktrace`` field (exception event) is always surfaced on stderr."""
    stdout, stderr = DataRobotWorkloadSandbox._partition_log_entries(
        [
            {"level": "INFO", "message": "before\n"},
            {
                "level": "ERROR",
                "message": "ValueError: bad",
                "stacktrace": "Traceback (most recent call last):\n  ...\nValueError: bad",
            },
        ]
    )
    assert stdout == "before"
    assert "ValueError: bad" in stderr
    assert "Traceback (most recent call last):" in stderr


def test_partition_log_entries_all_stdout_when_no_errors() -> None:
    """No regression: with only normal entries, stderr stays empty."""
    stdout, stderr = DataRobotWorkloadSandbox._partition_log_entries(
        [
            {"level": "INFO", "message": "a"},
            {"level": "DEBUG", "message": "b"},
            {"message": "c"},  # missing level defaults to stdout
        ]
    )
    assert stdout == "a\nb\nc"
    assert stderr == ""


def test_security_context_override_honored() -> None:
    ctx = SandboxSecurityContext(read_only_root_filesystem=False)
    sb = DataRobotWorkloadSandbox(
        image="dr-sandbox",
        datarobot_endpoint=API_BASE,
        datarobot_api_token="t",
        security_context=ctx,
    )
    payload = sb._build_workload_payload("x = 1", None, timeout_s=30.0)
    sc = payload["artifact"]["spec"]["containerGroups"][0]["containers"][0]["securityContext"]
    assert sc["readOnlyRootFilesystem"] is False


def test_create_payload_matches_workload_api_schema() -> None:
    """Lock in the Workload API contract: service artifact, a port on the primary
    container, and the resource signal carried as runtime resourceAllocation
    (matched to the artifact container by name) — not the old per-container
    resourceRequest / runtime.replicaCount, which the API now rejects.
    """
    payload = _sandbox()._build_workload_payload("_return = 1", None, timeout_s=30.0)

    assert payload["artifact"]["type"] == "service"
    group = payload["artifact"]["spec"]["containerGroups"][0]
    container = group["containers"][0]
    assert container["primary"] is True
    assert container["port"] >= 1024
    assert container.get("name")
    # The old fields the current Workload API rejects must be gone.
    assert "resourceRequest" not in container

    runtime_group = payload["runtime"]["containerGroups"][0]
    runtime_container = runtime_group["containers"][0]
    # Runtime container is matched to the artifact container by name.
    assert runtime_container["name"] == container["name"]
    assert runtime_group["name"] == group["name"]
    assert runtime_container["resourceAllocation"]["cpu"] >= 1
    assert "replicaCount" not in payload["runtime"]  # lives under the container group now
    assert runtime_group["replicaCount"] == 1
