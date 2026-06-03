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

from datarobot_genai.drtools.sandbox import DataRobotWorkloadSandbox
from datarobot_genai.drtools.sandbox import SandboxError
from datarobot_genai.drtools.sandbox import SandboxSecurityContext
from datarobot_genai.drtools.sandbox import SandboxTimeout

API_BASE = "https://app.datarobot.com/api/v2"
WORKLOAD_ID = "wkl_123"
CREATE_URL = f"{API_BASE}/console/workloads/"
GET_URL = f"{API_BASE}/console/workloads/{WORKLOAD_ID}"
DELETE_URL = f"{API_BASE}/console/workloads/{WORKLOAD_ID}"
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
