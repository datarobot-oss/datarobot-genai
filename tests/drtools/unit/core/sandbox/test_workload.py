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
import time

import httpx
import pytest
import respx

from datarobot_genai.drtools.core.sandbox import DataRobotWorkloadSandbox
from datarobot_genai.drtools.core.sandbox import SandboxError
from datarobot_genai.drtools.core.sandbox import SandboxSecurityContext
from datarobot_genai.drtools.core.sandbox import SandboxTimeout
from datarobot_genai.drtools.core.sandbox import workload as workload_mod

API_BASE = "https://app.datarobot.com/api/v2"
WORKLOAD_ID = "wkl_123"
CREATE_URL = f"{API_BASE}/workloads/"
GET_URL = f"{API_BASE}/workloads/{WORKLOAD_ID}"
DELETE_URL = f"{API_BASE}/workloads/{WORKLOAD_ID}"
LOGS_URL = f"{API_BASE}/otel/workload/{WORKLOAD_ID}/logs/"


def _sandbox(client: httpx.AsyncClient | None = None, **kwargs: object) -> DataRobotWorkloadSandbox:
    return DataRobotWorkloadSandbox(
        image="datarobotdev/datarobot-user-models:public_dropin_environments_dr_mcp_execute_sandbox_minimal_latest",
        datarobot_endpoint=API_BASE,
        datarobot_api_token="test-token",
        http_client=client,
        **kwargs,  # type: ignore[arg-type]
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


# Captured before any test monkeypatches ``asyncio.sleep`` so ``_noop_sleep``
# can still yield to the event loop without recursing into itself.
_REAL_SLEEP = asyncio.sleep


async def _noop_sleep(_seconds: float) -> None:
    """Collapse the poll backoff so flush-timing tests run instantly.

    The behaviour under test is *how many* polls are required before giving up,
    not the wall-clock between them.

    This still yields to the event loop (``sleep(0)``) rather than returning
    outright. ``run()`` drives the status poller and the log watcher as two
    concurrent tasks, so a sleep that never suspends lets the log watcher spin
    and *starve the status poller* -- ``watch.terminal`` would stay ``None`` for
    the whole test even though the mocked status endpoint reports a terminal
    state. That is a test-harness artifact with no production analogue, since a
    real ``asyncio.sleep(delay)`` always suspends.
    """
    await _REAL_SLEEP(0)


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
            # provisioning_timeout_s=0 so the tiny timeout_s alone drives the
            # poll deadline (the default allowance would stall the test).
            await _sandbox(client, provisioning_timeout_s=0.0).run("_return = 1", timeout_s=0.05)

    assert delete_route.called


@respx.mock
async def test_provisioning_does_not_consume_user_code_timeout() -> None:
    """Scheduling/image-pull time is covered by provisioning_timeout_s, not timeout_s.

    timeout_s is tiny; the workload sits in "provisioning" for several polls
    (longer than timeout_s alone would allow) before succeeding. With the
    provisioning allowance this must complete, not raise SandboxTimeout.
    """
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        side_effect=[
            httpx.Response(200, json={"id": WORKLOAD_ID, "status": "provisioning"}),
            httpx.Response(200, json={"id": WORKLOAD_ID, "status": "provisioning"}),
            httpx.Response(200, json={"id": WORKLOAD_ID, "status": "provisioning"}),
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
    respx.get(LOGS_URL).mock(return_value=_logs_response("__DR_SANDBOX_RESULT__:7"))
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        # Three "provisioning" polls take ~0.1+0.2+0.4s of backoff — well past
        # timeout_s=0.05 — so this run only succeeds because the provisioning
        # allowance extends the poll deadline.
        result = await _sandbox(client, provisioning_timeout_s=30.0).run(
            "_return = 7", timeout_s=0.05
        )

    assert result.return_value == 7


@respx.mock
async def test_errored_status_recovers_when_marker_flushes_late(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient ErrImagePull: workload goes terminal-errored before the
    container starts, and the result marker lands in the logs afterwards.

    The flush budget is shrunk to ~0 so the old fixed-budget behavior would
    give up ("no result marker in logs"); the failure-status path must extend
    the marker wait to the remaining overall budget and succeed.
    """
    monkeypatch.setattr(workload_mod, "_LOG_FLUSH_TIMEOUT_S", 0.001)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": WORKLOAD_ID,
                "status": "errored",
                "statusDetails": {"logTail": []},
            },
        )
    )
    respx.get(LOGS_URL).mock(
        side_effect=[
            _logs_response_entries(
                [
                    {
                        "timestamp": "2026-05-13T00:00:00Z",
                        "level": "ERROR",
                        "message": "lrs-x-sandbox: Image pull failed reason=ErrImagePull",
                    }
                ]
            ),
            # The OTEL endpoint returns the accumulated log set, so the
            # second read carries both the pull failure and the marker.
            _logs_response_entries(
                [
                    {
                        "timestamp": "2026-05-13T00:00:00Z",
                        "level": "ERROR",
                        "message": "lrs-x-sandbox: Image pull failed reason=ErrImagePull",
                    },
                    {
                        "timestamp": "2026-05-13T00:00:01Z",
                        "level": "INFO",
                        "message": "__DR_SANDBOX_RESULT__:99",
                    },
                ]
            ),
        ]
    )
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client, provisioning_timeout_s=30.0).run(
            "_return = 99", timeout_s=5.0
        )

    assert result.return_value == 99
    assert "ErrImagePull" in result.stderr


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


# --------------------------------------------------------------------------- #
# Partial OTEL flush must not be mistaken for "no marker"
# --------------------------------------------------------------------------- #


@respx.mock
async def test_partial_flush_on_failure_status_still_finds_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduces the staging false-failure.

    Observed sequence: a transient ErrImagePull marks the workload ``errored``
    while k8s retries; the container then runs to completion. OTEL delivers its
    stdout line by line, so the runner's own "RLIMIT_NPROC" line lands ~0.4s
    ahead of the result marker. Polling starts at 0.5s, so two consecutive reads
    saw only that first line and the wait was abandoned ~1s in — reporting "no
    result marker in logs" for a run that had actually succeeded.
    """
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"workloadId": WORKLOAD_ID, "status": "errored"})
    )
    partial = "sandbox process limit (RLIMIT_NPROC) set to 4096"
    complete = f"{partial}\n__DR_SANDBOX_RESULT__:99"
    # Same partial payload several times over, then the marker finally flushes.
    respx.get(LOGS_URL).mock(
        side_effect=[
            _logs_response(partial),
            _logs_response(partial),
            _logs_response(partial),
            _logs_response(partial),
            _logs_response(complete),
        ]
    )
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("code", timeout_s=5)

    assert result.return_value == 99
    assert "__DR_SANDBOX_RESULT__" not in result.stdout


@respx.mock
async def test_genuinely_markerless_failure_gives_up_on_the_quiet_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shortcut's purpose is preserved: a real crash must not wait out the budget.

    Output that arrives and then never changes exits via the quiet window, NOT
    the deadline — which matters because the failure-status budget is ~5
    minutes. Asserted by giving the deadline room (a large provisioning
    allowance) and still returning promptly.
    """
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    monkeypatch.setattr(workload_mod, "_STABLE_OUTPUT_NONEMPTY_QUIET_S", 0.05)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"workloadId": WORKLOAD_ID, "status": "errored"})
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response("Traceback: boom"))
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    started = time.monotonic()
    async with httpx.AsyncClient() as client:
        with pytest.raises(SandboxError):
            await _sandbox(client, provisioning_timeout_s=300.0).run("code", timeout_s=30)
    # Exited on the quiet window, nowhere near the 330s deadline.
    assert time.monotonic() - started < 30


@respx.mock
async def test_success_status_partial_flush_survives_one_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even on a success status, one repeated read is not proof the flush ended."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"workloadId": WORKLOAD_ID, "status": "succeeded"})
    )
    partial = "warming up"
    respx.get(LOGS_URL).mock(
        side_effect=[
            _logs_response(partial),
            _logs_response(partial),
            _logs_response(f"{partial}\n__DR_SANDBOX_RESULT__:7"),
        ]
    )
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("code", timeout_s=5)

    assert result.return_value == 7


@respx.mock
async def test_frozen_nonempty_stdout_still_waits_for_late_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the staging trace that survived the first fix.

    Captured from a real failing run: the runner's RLIMIT_NPROC line flushed and
    then stdout sat byte-identical at 97 bytes across three consecutive polls
    (~10s) while the marker was still being flushed. A quiet-window check on
    stdout gave up at 17s of a 324s budget; the endpoint had the marker 3s
    later. Non-empty output means the container is running and the runner prints
    its marker from after its own finally, so the wait must continue.
    """
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    # Deliberately SHORT non-empty budget: the point is that repetition alone
    # must not end the wait, not that the budget is large.
    monkeypatch.setattr(workload_mod, "_STABLE_OUTPUT_NONEMPTY_QUIET_S", 60.0)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"workloadId": WORKLOAD_ID, "status": "errored"})
    )
    frozen = "sandbox process limit (RLIMIT_NPROC) set to 4096"
    respx.get(LOGS_URL).mock(
        side_effect=[
            _logs_response(frozen),
            _logs_response(frozen),
            _logs_response(frozen),
            _logs_response(frozen),
            _logs_response(f"{frozen}\n__DR_SANDBOX_RESULT__:42"),
        ]
    )
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("code", timeout_s=5)

    assert result.return_value == 42


@respx.mock
async def test_empty_output_keeps_polling_for_a_late_starting_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty output must NOT be treated as "no marker coming".

    Regression for the review finding on #638. A transient ErrImagePull marks
    the workload errored *before* the runner starts; k8s retries and the
    container can begin a minute later, emitting its marker then. Throughout
    that window stdout is empty, because the pull's records are ERROR level and
    route to stderr. An early give-up on empty output therefore breaks exactly
    the case the provisioning budget exists for — so empty output has no
    stillness rule and polls until the deadline.
    """
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"workloadId": WORKLOAD_ID, "status": "errored"})
    )
    pull_err = {
        "timestamp": "t",
        "level": "ERROR",
        "message": "Image pull failed reason=ErrImagePull message=failed to pull and unpack image",
        "spanId": "s",
        "traceId": "t",
    }
    started_line = {
        "timestamp": "t",
        "level": "INFO",
        "message": "sandbox process limit (RLIMIT_NPROC) set to 4096",
        "spanId": "s",
        "traceId": "t",
    }
    marker = {
        "timestamp": "t",
        "level": "INFO",
        "message": "__DR_SANDBOX_RESULT__:99",
        "spanId": "s",
        "traceId": "t",
    }
    # Several polls where stdout is EMPTY (only the stderr-bound pull error),
    # then the retried container starts and produces its marker.
    respx.get(LOGS_URL).mock(
        side_effect=[
            _logs_response_entries([]),
            _logs_response_entries([pull_err]),
            _logs_response_entries([pull_err]),
            _logs_response_entries([pull_err]),
            _logs_response_entries([pull_err, started_line]),
            _logs_response_entries([pull_err, started_line, marker]),
        ]
    )
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("code", timeout_s=5)

    assert result.return_value == 99


@respx.mock
async def test_stderr_only_records_do_not_end_the_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ERROR-level records route to stderr, so they must not be read as stdout activity.

    In the captured trace a CrashLoopBackOff ERROR arrived while stdout stayed
    frozen — the stream was alive but `stdout` looked static.
    """
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    monkeypatch.setattr(workload_mod, "_STABLE_OUTPUT_NONEMPTY_QUIET_S", 60.0)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"workloadId": WORKLOAD_ID, "status": "errored"})
    )
    out = {"timestamp": "t", "level": "INFO", "message": "starting", "spanId": "s", "traceId": "t"}
    err = {
        "timestamp": "t",
        "level": "ERROR",
        "message": "Crash looping reason=CrashLoopBackOff",
        "spanId": "s",
        "traceId": "t",
    }
    done = {
        "timestamp": "t",
        "level": "INFO",
        "message": "__DR_SANDBOX_RESULT__:7",
        "spanId": "s",
        "traceId": "t",
    }
    respx.get(LOGS_URL).mock(
        side_effect=[
            _logs_response_entries([out]),
            _logs_response_entries([out, err]),
            _logs_response_entries([out, err]),
            _logs_response_entries([out, err, done]),
        ]
    )
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("code", timeout_s=5)

    assert result.return_value == 7


# --------------------------------------------------------------------------- #
# MODEL-24537: return at the result marker, not at terminal status
#
# Most tests below mock the status endpoint as permanently ``running`` /
# ``provisioning``. That is deliberate: a terminal-gated implementation would
# poll such a workload to its deadline and raise SandboxTimeout, so these tests
# only pass if the return is genuinely gated on the marker. On staging the wait
# they remove was 15-20s of a ~33s call.
# --------------------------------------------------------------------------- #


@respx.mock
async def test_marker_returns_without_waiting_for_terminal_status() -> None:
    """The headline behaviour: a marker in the logs ends the call.

    Status never leaves ``running``, so a terminal-gated implementation would
    poll until its deadline and raise SandboxTimeout. The marker is available
    immediately, so this must return promptly with the value.
    """
    respx.post(CREATE_URL).mock(return_value=_create_response())
    status_route = respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "running"})
    )
    # One empty read first, so the status poll gets requests in alongside us and
    # the concurrency is observable rather than short-circuited by an instant
    # marker.
    respx.get(LOGS_URL).mock(
        side_effect=[
            _logs_response(""),
            _logs_response("hi\n__DR_SANDBOX_RESULT__:42"),
        ]
    )
    delete_route = respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    started = time.monotonic()
    async with httpx.AsyncClient() as client:
        result = await _sandbox(client, provisioning_timeout_s=300.0).run(
            "_return = 42", timeout_s=30.0
        )
    elapsed = time.monotonic() - started

    assert result.return_value == 42
    assert result.stdout == "hi"
    # Nowhere near the 330s poll deadline a terminal-gated wait would have used.
    assert elapsed < 5.0
    # The status poll must NOT have disappeared — it ran concurrently the whole
    # time (repeatedly, on its own faster backoff), it just stopped being the gate.
    assert status_route.call_count > 1
    # Hazard 3: teardown still happens, and the marker was read before it.
    assert delete_route.called


@respx.mock
async def test_marker_first_return_reports_exit_code_zero() -> None:
    """Hazard 1: ``exit_code`` on a marker-first return.

    There is no terminal record to read ``exitCode`` from, and measured on
    staging that record reports ``exitCode: null`` for every sandbox workload
    anyway — so 0 is both what a caller already got and the only honest answer.
    Locked in so a future change has to think about it.
    """
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "running"})
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response("__DR_SANDBOX_RESULT__:1"))
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("_return = 1")

    assert result.exit_code == 0


@respx.mock
async def test_runner_timeout_detected_in_band_without_terminal_record() -> None:
    """Hazard 1: a user-code timeout still surfaces as SandboxTimeout.

    The runner exits 124 AND emits ``__DR_SANDBOX_RESULT__:null``, so the
    marker alone never means success. Without a terminal record the exit code
    is unavailable, so detection rides on the ``sandbox exceeded timeout of Ns``
    line the runner prints first.

    Captured from staging: that line arrives on the STDOUT side, because the
    OTEL endpoint tags the runner's own writes to fd 2 as INFO — only the
    platform's ``lrs-*`` records come through at ERROR severity.
    """
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "running"})
    )
    respx.get(LOGS_URL).mock(
        return_value=_logs_response_entries(
            [
                {"level": "INFO", "message": "sandbox process limit (RLIMIT_NPROC) set to 4096"},
                {"level": "INFO", "message": "sandbox exceeded timeout of 5s"},
                {"level": "INFO", "message": "__DR_SANDBOX_RESULT__:null"},
            ]
        )
    )
    delete_route = respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        with pytest.raises(SandboxTimeout, match="in-process timeout") as excinfo:
            await _sandbox(client).run("import time; time.sleep(30)", timeout_s=5.0)

    assert excinfo.value.exit_code == 124
    assert delete_route.called


@respx.mock
async def test_runner_timeout_detected_when_sentinel_lands_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same detection must work if the collector ever tags fd 2 as ERROR.

    Staging routes the sentinel to stdout today, but that is a collector
    mapping, not a contract — so the check spans stdout AND stderr.
    """
    monkeypatch.setattr(workload_mod, "_NULL_MARKER_RECHECK_S", 0.01)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "running"})
    )
    respx.get(LOGS_URL).mock(
        return_value=_logs_response_entries(
            [
                {"level": "ERROR", "message": "sandbox exceeded timeout of 5s"},
                {"level": "INFO", "message": "__DR_SANDBOX_RESULT__:null"},
            ]
        )
    )
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        with pytest.raises(SandboxTimeout, match="in-process timeout"):
            await _sandbox(client).run("import time; time.sleep(30)", timeout_s=5.0)


@respx.mock
async def test_null_marker_rechecks_logs_before_returning_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A null marker gets one confirming re-read, which can still find a timeout.

    Guards the batch-boundary case: the runner prints the sentinel and THEN the
    null marker, so a collector batch that splits the two would otherwise let a
    timed-out run return as a successful ``None``.
    """
    monkeypatch.setattr(workload_mod, "_NULL_MARKER_RECHECK_S", 0.01)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "running"})
    )
    logs_route = respx.get(LOGS_URL).mock(
        side_effect=[
            # Marker only — the sentinel has not been flushed yet.
            _logs_response("__DR_SANDBOX_RESULT__:null"),
            # The confirming re-read catches up with it.
            _logs_response("sandbox exceeded timeout of 5s\n__DR_SANDBOX_RESULT__:null"),
        ]
    )
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        with pytest.raises(SandboxTimeout, match="in-process timeout"):
            await _sandbox(client).run("import time; time.sleep(30)", timeout_s=5.0)

    assert logs_route.call_count == 2


@respx.mock
async def test_null_return_value_still_succeeds_after_the_recheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A snippet that legitimately returns nothing is not mistaken for a timeout."""
    monkeypatch.setattr(workload_mod, "_NULL_MARKER_RECHECK_S", 0.01)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "running"})
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response("all done\n__DR_SANDBOX_RESULT__:null"))
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("print('all done')")

    assert result.return_value is None
    assert result.stdout == "all done"
    assert result.exit_code == 0


@respx.mock
async def test_non_null_marker_pays_no_recheck_cost() -> None:
    """The confirming re-read is scoped to the ambiguous shape only.

    A real return value can never be the runner's timeout path (that path
    always emits null), so it must return on the first read.
    """
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "running"})
    )
    logs_route = respx.get(LOGS_URL).mock(
        return_value=_logs_response('__DR_SANDBOX_RESULT__:{"ok": true}')
    )
    respx_delete = respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("_return = {'ok': True}")

    assert result.return_value == {"ok": True}
    assert logs_route.call_count == 1
    assert respx_delete.called


@respx.mock
async def test_terminal_record_skips_the_recheck() -> None:
    """With a terminal record in hand the exit code is authoritative, so no re-read."""
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": WORKLOAD_ID,
                "status": "succeeded",
                "statusDetails": {"logTail": ["__DR_SANDBOX_RESULT__:null"]},
            },
        )
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response(""))
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    started = time.monotonic()
    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("print('x')")

    assert result.return_value is None
    # Would have cost _NULL_MARKER_RECHECK_S (1s, unpatched) had it re-read.
    assert time.monotonic() - started < 1.0


def test_duplicate_markers_from_restarts_resolve_deterministically() -> None:
    """Hazard 2: which marker instance gets parsed, given newest-first records.

    Verified against staging: the OTEL endpoint returns records NEWEST-FIRST
    (a timed-out run's stdout read back as attempt-2's lines, then attempt-1's),
    and ``_partition_log_entries`` joins them in that order. ``parse_result_marker``
    takes the LAST marker line, so it resolves to the OLDEST marker — the FIRST
    container attempt.

    That is safe rather than merely lucky: code and inputs are baked into the
    workload at create time, so every restart re-runs the identical snippet and
    there is no earlier run with a different result to inherit. This locks the
    resolution down so a change in either ordering has to be deliberate.
    """
    newest_first = [
        {"level": "INFO", "message": '__DR_SANDBOX_RESULT__:"attempt-2"'},
        {"level": "INFO", "message": "sandbox process limit (RLIMIT_NPROC) set to 4096"},
        {"level": "INFO", "message": '__DR_SANDBOX_RESULT__:"attempt-1"'},
        {"level": "INFO", "message": "sandbox process limit (RLIMIT_NPROC) set to 4096"},
    ]
    stdout, _ = DataRobotWorkloadSandbox._partition_log_entries(newest_first)
    clean, value = workload_mod.parse_result_marker(stdout)
    assert value == "attempt-1"
    # Only the resolved marker line is stripped; the duplicate stays visible in
    # stdout rather than being silently swallowed.
    assert '__DR_SANDBOX_RESULT__:"attempt-2"' in clean


@respx.mock
async def test_marker_first_return_sees_only_one_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning early narrows the duplicate-marker window to nothing.

    The restart markers observed on staging (2-3 per terminal-gated run) only
    accumulate while we keep waiting. Returning at the first marker means there
    is a single instance to resolve.
    """
    monkeypatch.setattr(workload_mod, "_LOG_POLL_INITIAL_DELAY_S", 0.01)
    monkeypatch.setattr(workload_mod, "_LOG_POLL_MAX_DELAY_S", 0.01)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "running"})
    )
    respx.get(LOGS_URL).mock(
        side_effect=[
            _logs_response("sandbox process limit (RLIMIT_NPROC) set to 4096"),
            _logs_response("__DR_SANDBOX_RESULT__:7\nsandbox process limit set"),
            # A restart's second marker — must never be reached.
            _logs_response("__DR_SANDBOX_RESULT__:7\n__DR_SANDBOX_RESULT__:7"),
        ]
    )
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("_return = 7")

    assert result.return_value == 7
    assert result.stdout.count("__DR_SANDBOX_RESULT__") == 0


@respx.mock
async def test_empty_output_keeps_polling_while_status_is_not_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hazard 4: the ErrImagePull-retry case survives the concurrency change.

    #638 made empty output poll to the deadline rather than give up, because a
    transient pull failure keeps stdout empty while k8s retries and the
    container can start a minute later. Now that log-watching begins at submit
    instead of at terminal status, that has to hold with NO terminal record at
    all — which is the common shape while a pull is retrying.
    """
    monkeypatch.setattr(workload_mod, "_LOG_POLL_INITIAL_DELAY_S", 0.01)
    monkeypatch.setattr(workload_mod, "_LOG_POLL_MAX_DELAY_S", 0.01)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "provisioning"})
    )
    pull_err = {"level": "ERROR", "message": "Image pull failed reason=ErrImagePull"}
    marker = {"level": "INFO", "message": "__DR_SANDBOX_RESULT__:99"}
    respx.get(LOGS_URL).mock(
        side_effect=[
            _logs_response_entries([]),
            _logs_response_entries([pull_err]),
            _logs_response_entries([pull_err]),
            _logs_response_entries([pull_err]),
            _logs_response_entries([pull_err]),
            _logs_response_entries([pull_err, marker]),
        ]
    )
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client, provisioning_timeout_s=30.0).run("code", timeout_s=5.0)

    assert result.return_value == 99
    assert "ErrImagePull" in result.stderr


@respx.mock
async def test_container_that_never_starts_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hazard 4: an unrecoverable failure must still fail, not return empty."""
    monkeypatch.setattr(workload_mod, "_LOG_FLUSH_TIMEOUT_S", 0.05)
    monkeypatch.setattr(workload_mod, "_LOG_POLL_INITIAL_DELAY_S", 0.01)
    monkeypatch.setattr(workload_mod, "_LOG_POLL_MAX_DELAY_S", 0.01)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "errored"})
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response_entries([]))
    delete_route = respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        with pytest.raises(SandboxError, match="no result marker"):
            await _sandbox(client, provisioning_timeout_s=0.0).run("code", timeout_s=0.05)

    assert delete_route.called


@respx.mock
async def test_status_poll_failure_is_not_fatal_when_the_marker_arrives() -> None:
    """The marker outranks a broken status endpoint.

    Status is no longer the gate, so a 500 from it must not fail a run whose
    snippet demonstrably completed. Previously this aborted before the logs
    were ever read.
    """
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(return_value=httpx.Response(500, text="upstream boom"))
    respx.get(LOGS_URL).mock(return_value=_logs_response("__DR_SANDBOX_RESULT__:5"))
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("_return = 5")

    assert result.return_value == 5


@respx.mock
async def test_status_poll_failure_is_fatal_when_no_marker_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """...but with nothing from either source, the status error is the real one."""
    monkeypatch.setattr(workload_mod, "_LOG_POLL_INITIAL_DELAY_S", 0.01)
    monkeypatch.setattr(workload_mod, "_LOG_POLL_MAX_DELAY_S", 0.01)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(return_value=httpx.Response(500, text="upstream boom"))
    respx.get(LOGS_URL).mock(return_value=_logs_response_entries([]))
    delete_route = respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        with pytest.raises(SandboxError, match="status fetch failed"):
            await _sandbox(client, provisioning_timeout_s=0.0).run("code", timeout_s=0.05)

    assert delete_route.called


@respx.mock
async def test_terminal_timeout_status_does_not_burn_the_flush_budget() -> None:
    """A terminal ``timeout`` status raises regardless of the logs, so stop waiting.

    Locks in the short-circuit: the outcome cannot change, so spending
    _LOG_FLUSH_TIMEOUT_S looking for a marker is pure latency.
    """
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "timeout"})
    )
    respx.get(LOGS_URL).mock(return_value=_logs_response_entries([]))
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    started = time.monotonic()
    async with httpx.AsyncClient() as client:
        with pytest.raises(SandboxTimeout):
            await _sandbox(client).run("_return = 1")

    assert time.monotonic() - started < workload_mod._LOG_FLUSH_TIMEOUT_S / 2


@respx.mock
async def test_cancellation_leaves_no_pending_status_task() -> None:
    """Hazard 5: both concurrent tasks are cancelled cleanly."""
    respx.post(CREATE_URL).mock(return_value=_create_response())
    logs_started = asyncio.Event()

    async def _slow_logs(request: httpx.Request) -> httpx.Response:
        logs_started.set()
        await asyncio.sleep(5.0)
        return _logs_response("")

    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"id": WORKLOAD_ID, "status": "running"})
    )
    respx.get(LOGS_URL).mock(side_effect=_slow_logs)
    delete_route = respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        task = asyncio.create_task(_sandbox(client).run("_return = 1", timeout_s=30.0))
        await logs_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    # Give the loop a turn to finish reaping anything that was cancelled.
    await asyncio.sleep(0)
    leftover = [
        t
        for t in asyncio.all_tasks()
        if t.get_name().startswith("dr-sandbox-status-") and not t.done()
    ]
    assert leftover == []
    assert delete_route.called


@respx.mock
async def test_quiet_snippet_is_not_abandoned_before_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the review finding on #658.

    The log watcher now starts at submit, and the image runner prints a startup
    line BEFORE user code. So stdout goes non-empty immediately and then sits
    unchanged for as long as the snippet runs quietly. The stillness rule must
    not fire in that window, or a still-running container is abandoned and
    `timeout_s` stops bounding user code once that first line appears.

    Here the workload never reaches a terminal status while stdout is frozen at
    the startup line for many polls; the marker only arrives afterwards.
    """
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    # Deliberately tiny: repetition alone must not end the wait pre-terminal.
    monkeypatch.setattr(workload_mod, "_STABLE_OUTPUT_NONEMPTY_QUIET_S", 0.01)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    # Status stays non-terminal for the whole frozen-stdout window.
    respx.get(GET_URL).mock(
        side_effect=[
            httpx.Response(200, json={"workloadId": WORKLOAD_ID, "status": "running"}),
            httpx.Response(200, json={"workloadId": WORKLOAD_ID, "status": "running"}),
            httpx.Response(200, json={"workloadId": WORKLOAD_ID, "status": "running"}),
            httpx.Response(200, json={"workloadId": WORKLOAD_ID, "status": "running"}),
            httpx.Response(200, json={"workloadId": WORKLOAD_ID, "status": "errored"}),
        ]
    )
    startup = "sandbox process limit (RLIMIT_NPROC) set to 4096"
    respx.get(LOGS_URL).mock(
        side_effect=[
            _logs_response(startup),
            _logs_response(startup),
            _logs_response(startup),
            _logs_response(startup),
            _logs_response(startup),
            _logs_response(f'{startup}\n__DR_SANDBOX_RESULT__:{{"slow": true}}'),
        ]
    )
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("code", timeout_s=30)

    assert result.return_value == {"slow": True}


@respx.mock
async def test_non_json_logs_response_does_not_fail_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 that is not the documented JSON shape must be retried, not raised.

    `_read_logs_once` documents that it never raises, because the logs endpoint
    is the source of truth for the result and one bad read must not fail an
    execution. Regression for the review finding that `resp.json()` and
    `_partition_log_entries` sat outside its try.
    """
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    respx.post(CREATE_URL).mock(return_value=_create_response())
    respx.get(GET_URL).mock(
        return_value=httpx.Response(200, json={"workloadId": WORKLOAD_ID, "status": "errored"})
    )
    respx.get(LOGS_URL).mock(
        side_effect=[
            httpx.Response(200, text="<html>gateway</html>"),  # not JSON at all
            httpx.Response(200, json={"data": "not-a-list"}),  # wrong shape
            _logs_response("__DR_SANDBOX_RESULT__:5"),  # then the real thing
        ]
    )
    respx.delete(DELETE_URL).mock(return_value=httpx.Response(204))

    async with httpx.AsyncClient() as client:
        result = await _sandbox(client).run("code", timeout_s=5)

    assert result.return_value == 5
