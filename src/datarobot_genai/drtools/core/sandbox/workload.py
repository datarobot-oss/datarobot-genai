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

"""DataRobot workload-api backed sandbox implementation.

Submits a single-container workload to the DataRobot workload-api console
endpoints, polls the workload until terminal, fetches container output
(stdout + stderr, split by OTEL severity) from the OTEL log endpoint, and
parses the ``__DR_SANDBOX_RESULT__:`` marker
(see :mod:`datarobot_genai.drtools.core.sandbox.protocol`) emitted by the
container runner shipped in the sandbox image (datarobot-user-models#2137).

Endpoint surface
----------------
- ``POST /api/v2/console/workloads/`` — create a workload (inline artifact).
- ``GET  /api/v2/console/workloads/{id}`` — poll status.
- ``DELETE /api/v2/console/workloads/{id}`` — forced teardown (always
  fired from ``finally`` for success / failure / timeout / cancellation).
- ``GET  /api/v2/otel/workload/{id}/logs/`` — container stdout via the
  OTEL collector. Trailing slash is required; some httpx setups don't
  follow the 308 redirect emitted when omitted.

The container's stdout (written to ``/dev/stdout`` by the image's runner,
datarobot-user-models#2137) is picked up by the OTEL collector and surfaces
here. ``statusDetails.logTail`` from the terminal
workload response is used as a secondary source in case the OTEL pipeline
hasn't flushed by the time the workload reaches a terminal state.
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any
from urllib.parse import urljoin

import httpx

from datarobot_genai.drtools.core.sandbox.base import SandboxError
from datarobot_genai.drtools.core.sandbox.base import SandboxResult
from datarobot_genai.drtools.core.sandbox.base import SandboxSecurityContext
from datarobot_genai.drtools.core.sandbox.base import SandboxTimeout
from datarobot_genai.drtools.core.sandbox.protocol import SANDBOX_TIMEOUT_EXIT_CODE
from datarobot_genai.drtools.core.sandbox.protocol import parse_result_marker

logger = logging.getLogger(__name__)

_TERMINAL_SUCCESS = {"succeeded", "completed"}
_TERMINAL_FAILURE = {"failed", "errored", "stopped"}
_TERMINAL_TIMEOUT = {"timeout", "timedout", "timed_out"}
_TERMINAL_STATES = _TERMINAL_SUCCESS | _TERMINAL_FAILURE | _TERMINAL_TIMEOUT

# OTEL log severity levels we route to stderr. The DataRobot OTEL logs endpoint
# (datavolt_to_dr_otel_log) returns no stdout/stderr stream attribute — only a
# severity `level` and an optional `stacktrace`. The container runner
# (datarobot-user-models#2137) writes its result marker + user prints to stdout
# but emits tracebacks / diagnostics to stderr, which the collector surfaces at
# these severities. Anything else (incl. the INFO result-marker line) stays in
# stdout so marker parsing is unaffected.
_STDERR_LEVELS = {"ERROR", "CRITICAL", "FATAL"}

# Short, bounded timeout for best-effort DELETE in finally / cancellation
# paths. We don't want a hung teardown to block caller cancellation.
_TEARDOWN_TIMEOUT_S = 5.0

# Port the sandbox runner image serves on. The workload-api requires a port on
# primary (service) containers and enforces >= 1024.
_SANDBOX_RUNNER_PORT = 8080

# Names tying the runtime container to its artifact container (matched by name).
_SANDBOX_GROUP_NAME = "default"
_SANDBOX_CONTAINER_NAME = "sandbox"


class DataRobotWorkloadSandbox:
    """Sandbox implementation backed by the DataRobot workload-api.

    Submits a single-container workload running the configured ``image``
    with the user code and inputs base64-encoded in env vars
    (``DR_SANDBOX_CODE_B64`` / ``DR_SANDBOX_INPUTS_B64``). Polls workload
    status with exponential backoff (capped at 2s) until terminal, fetches
    container output from the OTEL log endpoint (splitting stdout from
    stderr by log severity), and parses the final
    ``__DR_SANDBOX_RESULT__:`` line on stdout for the return value.

    The workload is always deleted in a ``finally`` block (success,
    failure, timeout, cancellation), so callers don't leave orphan
    workloads behind.

    Parameters
    ----------
    image
        Container image URI for the sandbox runner. Defaults to the image
        built by DRUM from ``public_dropin_environments/dr_mcp_execute_sandbox_minimal``
        (datarobot/datarobot-user-models#2137):
        ``datarobotdev/datarobot-user-models:public_dropin_environments_dr_mcp_execute_sandbox_minimal_latest``.
    datarobot_endpoint
        DataRobot API endpoint (e.g.
        ``https://app.datarobot.com/api/v2``). Used to derive the
        workload-api base when ``workload_api_base`` is not provided.
    datarobot_api_token
        Bearer token sent as ``Authorization: Bearer <token>``.
    security_context
        Container security context. When ``None`` (the default) the
        ``securityContext`` field is omitted from the workload payload and
        the workload-api applies cluster defaults. Sending a tightened
        context requires the ``WORKLOAD_API_SECURITY_CONTEXT`` entitlement
        (datarobot/DataRobot#153183) — callers should gate construction of
        a non-None value on that flag.
    workload_api_base
        Optional override for the workload-api base URL. Falls back to the
        DataRobot endpoint host with ``/api/v2`` appended.
    http_client
        Optional :class:`httpx.AsyncClient` for dependency injection /
        testing. When ``None``, a client is created and closed per call.

    Notes
    -----
    ``externals`` injection (for CodeMode-style tool passthrough) is a
    follow-up tied to tool-catalog serialization. Passing a non-empty
    ``externals`` mapping raises :class:`NotImplementedError`.
    """

    def __init__(
        self,
        image: str,
        datarobot_endpoint: str,
        datarobot_api_token: str,
        security_context: SandboxSecurityContext | None = None,
        workload_api_base: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.image = image
        self.datarobot_endpoint = datarobot_endpoint.rstrip("/")
        self.datarobot_api_token = datarobot_api_token
        self.security_context = security_context
        self.workload_api_base = (
            workload_api_base.rstrip("/")
            if workload_api_base is not None
            else self._derive_workload_base(self.datarobot_endpoint)
        )
        self._http_client = http_client

    @staticmethod
    def _derive_workload_base(endpoint: str) -> str:
        if endpoint.endswith("/api/v2"):
            return endpoint
        return f"{endpoint}/api/v2"

    def _build_workload_payload(
        self,
        code: str,
        inputs: dict[str, Any] | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
        inputs_b64 = base64.b64encode(json.dumps(inputs or {}).encode("utf-8")).decode("ascii")
        run_id = uuid.uuid4().hex[:12]
        # Pass our caller-side wall-clock through to the runner so its
        # in-process SIGALRM cap aligns with what we'll enforce here. The
        # runner exits 124 + emits a null result marker if its cap fires
        # first; either way we surface as SandboxTimeout. Floor of 1s so
        # the runner doesn't see 0 (which means "disabled" inside runner).
        runner_timeout = max(1, int(timeout_s))
        container: dict[str, Any] = {
            "name": _SANDBOX_CONTAINER_NAME,
            "imageUri": self.image,
            "primary": True,
            # Primary (service) containers must declare a port >= 1024. The
            # runner is a one-shot job (runs the snippet, emits a result marker
            # to stdout, exits); we read its result from the OTEL logs, so the
            # port is only here to satisfy the service schema.
            "port": _SANDBOX_RUNNER_PORT,
            "environmentVars": [
                {"name": "DR_SANDBOX_CODE_B64", "value": code_b64},
                {"name": "DR_SANDBOX_INPUTS_B64", "value": inputs_b64},
                {"name": "DR_SANDBOX_TIMEOUT_SECS", "value": str(runner_timeout)},
            ],
        }
        if self.security_context is not None:
            container["securityContext"] = self.security_context.to_workload_api_dict()
        # The artifact half describes the container (image/port/env); the runtime
        # half carries deployment knobs (replicas + per-container resources),
        # matched to the artifact container by name. See the Workload API docs.
        return {
            "name": f"dr-sandbox-{run_id}",
            "artifact": {
                "name": f"dr-sandbox-artifact-{run_id}",
                "type": "service",
                "spec": {
                    "containerGroups": [{"name": _SANDBOX_GROUP_NAME, "containers": [container]}]
                },
            },
            "runtime": {
                "containerGroups": [
                    {
                        "name": _SANDBOX_GROUP_NAME,
                        "replicaCount": 1,
                        "containers": [
                            {
                                "name": _SANDBOX_CONTAINER_NAME,
                                "resourceAllocation": {"cpu": 1, "memory": 536870912},
                            }
                        ],
                    }
                ]
            },
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.datarobot_api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _workloads_url(self, suffix: str = "") -> str:
        # POST to create needs a trailing slash; GET/DELETE by id does not.
        base = urljoin(self.workload_api_base + "/", "workloads/")
        if not suffix:
            return base
        return base + suffix.lstrip("/")

    def _logs_url(self, workload_id: str) -> str:
        # Trailing slash REQUIRED — the OTEL endpoint 308-redirects
        # without it and not every httpx setup follows redirects.
        return urljoin(
            self.workload_api_base + "/",
            f"otel/workload/{workload_id}/logs/",
        )

    async def _submit(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> str:
        resp = await client.post(
            self._workloads_url(),
            json=payload,
            headers=self._headers(),
        )
        if resp.status_code >= 400:
            raise SandboxError(f"workload-api create failed: {resp.status_code} {resp.text}")
        body = resp.json()
        # Slack examples show a top-level ``workloadId`` in the create
        # response (e.g. Abdo's seismogemma-training-artifact). Fall back
        # to the bare ``id`` field on the WorkloadFormatted schema.
        workload_id = body.get("workloadId") or body.get("id")
        if not workload_id:
            raise SandboxError(f"workload-api response missing workload id: {body!r}")
        return str(workload_id)

    async def _poll(
        self,
        client: httpx.AsyncClient,
        workload_id: str,
        deadline: float,
    ) -> dict[str, Any]:
        delay = 0.1
        while True:
            resp = await client.get(
                self._workloads_url(workload_id),
                headers=self._headers(),
            )
            if resp.status_code >= 400:
                raise SandboxError(
                    f"workload-api status fetch failed: {resp.status_code} {resp.text}"
                )
            body = resp.json()
            status = str(body.get("status", "")).lower()
            if status in _TERMINAL_STATES:
                return body
            if time.monotonic() > deadline:
                raise SandboxTimeout(
                    f"sandbox exceeded timeout while polling workload {workload_id}"
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 2.0)

    @staticmethod
    def _partition_log_entries(data: list[dict[str, Any]]) -> tuple[str, str]:
        """Split OTEL log entries into ``(stdout, stderr)`` by severity.

        Each entry has the shape produced by ``datavolt_to_dr_otel_log``:
        ``{"level", "message", "stacktrace"?, ...}``. There is no
        stdout/stderr stream attribute, so we route by severity: entries
        at :data:`_STDERR_LEVELS` (and any ``stacktrace`` text) go to
        stderr; everything else — including the INFO result-marker line —
        stays in stdout so :func:`parse_result_marker` still sees it.
        """
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        for entry in data:
            message = str(entry.get("message", ""))
            level = str(entry.get("level", "")).upper()
            if level in _STDERR_LEVELS:
                stderr_parts.append(message)
            else:
                stdout_parts.append(message)
            # Exception events carry the traceback in a separate field that
            # would otherwise be dropped; always surface it on stderr.
            stacktrace = entry.get("stacktrace")
            if stacktrace:
                stderr_parts.append(str(stacktrace))
        return "".join(stdout_parts), "".join(stderr_parts)

    async def _fetch_logs(
        self,
        client: httpx.AsyncClient,
        workload_id: str,
        terminal: dict[str, Any],
    ) -> tuple[str, str]:
        """Return ``(stdout, stderr)``, preferring OTEL logs over ``logTail``.

        The OTEL endpoint returns the paginated shape
        ``{"count", "next", "previous", "data": [{"message", "level", ...}, ...]}``.
        Entries are partitioned into stdout/stderr by severity (see
        :meth:`_partition_log_entries`). If the OTEL pipeline hasn't flushed
        yet (empty data), we fall back to the ``statusDetails.logTail`` array
        from the terminal workload response (treated as stdout, since logTail
        carries no per-line severity).
        """
        stdout = ""
        stderr = ""
        try:
            resp = await client.get(
                self._logs_url(workload_id),
                headers=self._headers(),
            )
            if resp.status_code < 400:
                body = resp.json()
                data = body.get("data") or []
                stdout, stderr = self._partition_log_entries(data)
            else:
                logger.warning(
                    "workload-api logs fetch failed: %s %s",
                    resp.status_code,
                    resp.text,
                )
        except Exception:  # pragma: no cover — defensive
            logger.exception("workload-api logs fetch raised; falling back to logTail")

        if stdout.strip() or stderr.strip():
            return stdout, stderr

        # Fallback: the workload response may carry a small tail of
        # container output in statusDetails.logTail for cases where the
        # OTEL pipeline hasn't flushed yet. logTail has no per-line
        # severity, so it all surfaces as stdout.
        log_tail = (terminal.get("statusDetails") or {}).get("logTail") or []
        if log_tail:
            tail_parts: list[str] = []
            for entry in log_tail:
                if isinstance(entry, str):
                    tail_parts.append(entry)
                elif isinstance(entry, dict):
                    tail_parts.append(str(entry.get("message", "")))
            stdout = "\n".join(p for p in tail_parts if p)
        return stdout, stderr

    async def _delete(self, workload_id: str) -> None:
        """Best-effort DELETE; swallow all errors and never raise."""
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(_TEARDOWN_TIMEOUT_S)
            ) as teardown_client:
                resp = await teardown_client.delete(
                    self._workloads_url(workload_id),
                    headers=self._headers(),
                )
                if resp.status_code >= 400 and resp.status_code != 404:
                    logger.warning(
                        "workload-api delete returned %s for %s: %s",
                        resp.status_code,
                        workload_id,
                        resp.text,
                    )
        except Exception:
            logger.exception("workload-api delete raised for workload %s; ignoring", workload_id)

    async def run(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        externals: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> SandboxResult:
        """Execute ``code`` in a workload-api container. See :class:`Sandbox.run`."""
        if externals:
            raise NotImplementedError(
                "externals injection is not yet supported by "
                "DataRobotWorkloadSandbox; tracked as a follow-up tied to "
                "CodeMode tool-catalog serialization."
            )

        payload = self._build_workload_payload(code, inputs, timeout_s)
        start = time.monotonic()
        deadline = start + timeout_s

        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s + 5))
        workload_id: str | None = None
        try:
            workload_id = await self._submit(client, payload)
            terminal = await self._poll(client, workload_id, deadline)
            stdout_raw, stderr = await self._fetch_logs(client, workload_id, terminal)
        except asyncio.CancelledError:
            # Make sure cleanup still fires even when the caller cancels
            # our task. The finally below will run; we just need to
            # re-raise after it completes.
            raise
        finally:
            if owns_client:
                await client.aclose()
            if workload_id is not None:
                await self._delete(workload_id)

        stdout, return_value = parse_result_marker(stdout_raw)
        duration = time.monotonic() - start
        status = str(terminal.get("status", "")).lower()
        exit_code = int(terminal.get("exitCode", 0) or 0)

        if status in _TERMINAL_TIMEOUT:
            raise SandboxTimeout(f"workload-api workload {workload_id} timed out: {terminal!r}")
        # The runner exits SANDBOX_TIMEOUT_EXIT_CODE (124) when its
        # in-process SIGALRM cap fires before the caller / workload-api
        # cap. Surface as SandboxTimeout so callers see one unified
        # timeout path regardless of which layer tripped first. See
        # datarobot/datarobot-user-models#2137 for the runner-side cap.
        if exit_code == SANDBOX_TIMEOUT_EXIT_CODE:
            raise SandboxTimeout(
                f"workload-api workload {workload_id} runner exceeded its "
                f"in-process timeout (exit {SANDBOX_TIMEOUT_EXIT_CODE}); "
                f"caller timeout_s={timeout_s}"
            )
        if status in _TERMINAL_FAILURE:
            raise SandboxError(f"workload-api workload {workload_id} failed: status={status}")

        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            return_value=return_value,
            duration_s=duration,
            exit_code=exit_code,
        )
