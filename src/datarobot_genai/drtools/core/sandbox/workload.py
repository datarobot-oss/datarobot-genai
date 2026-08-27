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
endpoints, watches container output (stdout + stderr, split by OTEL severity)
from the OTEL log endpoint *concurrently* with workload status, and returns as
soon as the ``__DR_SANDBOX_RESULT__:`` marker
(see :mod:`datarobot_genai.drtools.core.sandbox.protocol`) emitted by the
container runner shipped in the sandbox image (datarobot-user-models#2137) is
available.

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

Why the marker gates the return, and not the status (MODEL-24537)
----------------------------------------------------------------
The runner is a ONE-SHOT job, but the workload-api only schedules long-running
"service" (or "nim") artifacts. So the container runs the snippet in well under
a second, prints its marker, and exits 0 — and Kubernetes treats an exit-0
service container as a crash and restarts it
(``Running (recovered from termination) last_reason=Completed
last_exit_code=0``). It runs again, emits the marker again, and only after
repeated restarts does k8s declare ``CrashLoopBackOff`` (``back-off 10s``, then
``back-off 20s``) and the workload-api finally report ``errored``.

Gating the return on a terminal status therefore meant waiting for Kubernetes
to give up on a container that had already done its job. Measured against
staging, the marker was available at a mean of 18.3s while terminal status
arrived at 32.8s — 30-54% of every call spent waiting for a decision that could
not change the answer. So the status poll still runs (its terminal record is
the only place a ``timeout`` status or ``logTail`` fallback comes from, and it
is what fails a container that never starts), but it is no longer the gate.

Consequences that had to be handled, and how:

exit code
    The workload record's ``exitCode`` is the only place a container exit
    status could come from, and returning early means not having a terminal
    record. Measured on staging, that record reports ``exitCode: null`` for
    every sandbox workload — it describes a *service*, not a job — so
    ``exit_code`` was already always 0 here and a marker-first return loses
    nothing. That also means the ``exitCode == 124`` timeout check could never
    have fired on this backend; the runner's ``sandbox exceeded timeout of Ns``
    stderr line (:func:`has_timeout_marker`) is now the signal that surfaces a
    user-code timeout, and the exit-code check is kept only for backends /
    future API versions that do populate it.

null marker ambiguity
    The runner emits its marker from a ``finally``, so EVERY path that starts
    produces one — including its own timeout path, which emits
    ``__DR_SANDBOX_RESULT__:null``. A marker alone is therefore never proof of
    success. A null payload with no timeout sentinel yet in hand triggers one
    short confirming re-read (:data:`_NULL_MARKER_RECHECK_S`) before the run is
    allowed to return successfully.

multiple markers
    Restarts emit the marker once per container attempt. This is safe because
    the code and inputs are baked into the workload at create time, so every
    attempt runs the identical snippet — there is no "previous run with a
    different result" to pick up. Returning at the FIRST marker observed also
    strictly narrows the window in which duplicates can accumulate: measured
    on staging, a terminal-gated run had 2-3 markers to choose between.
"""

import asyncio
import base64
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from urllib.parse import urljoin

import httpx

from datarobot_genai.drtools.core.sandbox.base import SandboxError
from datarobot_genai.drtools.core.sandbox.base import SandboxResult
from datarobot_genai.drtools.core.sandbox.base import SandboxSecurityContext
from datarobot_genai.drtools.core.sandbox.base import SandboxTimeout
from datarobot_genai.drtools.core.sandbox.protocol import SANDBOX_TIMEOUT_EXIT_CODE
from datarobot_genai.drtools.core.sandbox.protocol import has_result_marker
from datarobot_genai.drtools.core.sandbox.protocol import has_timeout_marker
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

# The workload can hit a terminal status (a one-shot runner exits, so the
# workload-api flags the "service" errored) before the OTEL collector flushes
# the container's stdout — including the result marker. Poll the logs endpoint
# up to this budget for the marker rather than reading once and racing the flush.
_LOG_FLUSH_TIMEOUT_S = 30.0

# How long NON-EMPTY output must stay unchanged before we conclude no result
# marker is coming.
#
# Only non-empty output gets a stillness rule. Once the container has printed
# anything, a marker is in flight: the image runner prints it AFTER its own
# try/except BaseException/finally (datarobot-user-models,
# dr_mcp_execute_sandbox_minimal/runner.py), so every container that starts
# emits one on every path — normal return, user exception, SystemExit, and the
# in-process timeout. How long the OTEL collector then takes to flush it is not
# boundable from here, so wait, and give up only after a long stillness. That
# covers the one genuine no-marker-with-output case: a container SIGKILLed
# before reaching the print.
#
# EMPTY output deliberately has NO stillness rule and polls to the deadline.
# It cannot be distinguished from "the container has not started yet": a
# transient ErrImagePull marks the workload errored before the runner starts,
# k8s retries, and the container can begin a minute later — and the pull's
# ERROR records route to stderr, so stdout stays empty throughout. Giving up
# early on empty output would break exactly the case the provisioning budget
# exists for.
_STABLE_OUTPUT_NONEMPTY_QUIET_S = 30.0

# Grace spent re-reading the logs before a NULL result marker is allowed to
# return successfully on the marker-first path.
#
# The runner's timeout path prints "sandbox exceeded timeout of Ns" to stderr
# and THEN emits `__DR_SANDBOX_RESULT__:null` — same process, that order, so
# the accumulated log read that carries the marker essentially always carries
# the sentinel too. "Essentially always" is not "always": a collector batch
# boundary landing between the two lines would let a timed-out run return as a
# successful `None`. Since a marker-first return has no terminal record to
# check an exit code against, spend one short re-read on the exact ambiguous
# shape (null payload, no sentinel seen, no terminal record) and nothing else.
_NULL_MARKER_RECHECK_S = 1.0

# Log-poll backoff: first interval and its cap. The logs endpoint is now polled
# from submit rather than from terminal status, so the cap keeps a long
# provisioning wait from turning into a request flood.
_LOG_POLL_INITIAL_DELAY_S = 0.5
_LOG_POLL_MAX_DELAY_S = 3.0


# Provisioning allowance added on top of the caller's ``timeout_s`` when
# computing the status-poll deadline. ``timeout_s`` bounds USER CODE, and is
# already enforced inside the container (DR_SANDBOX_TIMEOUT_SECS → SIGALRM →
# exit 124); scheduling and image pulls are infrastructure time and must not
# eat into it. A cold node pulling the sandbox image for the first time
# routinely takes 60–90s — longer than the default 30s ``timeout_s`` — which
# used to surface as SandboxTimeout before user code ever ran. This allowance
# bounds only how long we wait for the infra side; genuinely hung workloads
# still fail, just later.
_DEFAULT_PROVISIONING_TIMEOUT_S = 300.0

# Port the sandbox runner image serves on. The workload-api requires a port on
# primary (service) containers and enforces >= 1024.
_SANDBOX_RUNNER_PORT = 8080

# Names tying the runtime container to its artifact container (matched by name).
_SANDBOX_GROUP_NAME = "default"
_SANDBOX_CONTAINER_NAME = "sandbox"


@dataclass
class _StatusWatch:
    """Handoff from the concurrent status poll to the log watcher.

    The status poll no longer gates the return, so it writes what it learns
    here instead of being awaited for it:

    ``terminal``
        The terminal workload record, once one exists. Source of the
        ``timeout`` status, ``statusDetails.logTail``, and ``exitCode``.
    ``terminal_at``
        When it arrived — the marker wait's flush budget is measured from this,
        not from submit.
    ``error``
        Anything :meth:`DataRobotWorkloadSandbox._poll` raised. Deliberately
        NOT fatal on its own: a result marker outranks a status-fetch failure
        (the marker proves the snippet ran), so this is only re-raised when no
        marker was found either.
    """

    terminal: dict[str, Any] | None = None
    terminal_at: float | None = None
    error: BaseException | None = field(default=None, repr=False)

    @property
    def status(self) -> str:
        return str((self.terminal or {}).get("status", "")).lower()


class DataRobotWorkloadSandbox:
    """Sandbox implementation backed by the DataRobot workload-api.

    Submits a single-container workload running the configured ``image``
    with the user code and inputs base64-encoded in env vars
    (``DR_SANDBOX_CODE_B64`` / ``DR_SANDBOX_INPUTS_B64``). Watches the OTEL log
    endpoint (splitting stdout from stderr by log severity) CONCURRENTLY with
    workload status, and returns as soon as the ``__DR_SANDBOX_RESULT__:`` line
    is available on stdout — not when the workload reaches a terminal status,
    which on this backend means waiting out a Kubernetes CrashLoopBackOff for a
    one-shot container that already finished (MODEL-24537; see the module
    docstring). Status is still polled with exponential backoff (capped at 2s):
    it is what fails a container that never starts, and the only source of a
    terminal ``timeout`` status or the ``logTail`` fallback.

    The workload is always deleted in a ``finally`` block (success,
    failure, timeout, cancellation), so callers don't leave orphan
    workloads behind — and the result marker is always read before the
    delete fires.

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
    provisioning_timeout_s
        Infrastructure allowance added on top of ``run()``'s ``timeout_s``
        when waiting for the workload to reach a terminal status. Covers
        scheduling and image pulls (a cold node's first pull of the sandbox
        image takes 60–90s), which are not the user code's time to spend —
        the in-container runner enforces ``timeout_s`` on the code itself.
        Defaults to :data:`_DEFAULT_PROVISIONING_TIMEOUT_S`.

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
        provisioning_timeout_s: float = _DEFAULT_PROVISIONING_TIMEOUT_S,
    ) -> None:
        self.image = image
        self.provisioning_timeout_s = provisioning_timeout_s
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
        # Join entries with newlines so the result marker survives as its own
        # line for parse_result_marker (splitlines) — concatenating with "" mashes
        # it into adjacent lines and breaks JSON decoding. Strip each entry's own
        # trailing newline first so entries that already end in "\n" don't produce
        # blank lines.
        return (
            "\n".join(p.rstrip("\n") for p in stdout_parts),
            "\n".join(p.rstrip("\n") for p in stderr_parts),
        )

    @staticmethod
    def _extract_log_tail(terminal: dict[str, Any]) -> str:
        """Flatten ``statusDetails.logTail`` (str or ``{message}`` entries) to text."""
        log_tail = (terminal.get("statusDetails") or {}).get("logTail") or []
        parts: list[str] = []
        for entry in log_tail:
            if isinstance(entry, str):
                parts.append(entry)
            elif isinstance(entry, dict):
                parts.append(str(entry.get("message", "")))
        return "\n".join(p for p in parts if p)

    async def _read_logs_once(
        self,
        client: httpx.AsyncClient,
        workload_id: str,
    ) -> tuple[str, str]:
        """One read of the OTEL logs endpoint as ``(stdout, stderr)``.

        The endpoint returns the paginated shape
        ``{"count", "next", "previous", "data": [{"message", "level", ...}, ...]}``
        and always carries the run's ACCUMULATED records, so a single read is a
        full snapshot rather than an increment. Entries are partitioned into
        stdout/stderr by severity (see :meth:`_partition_log_entries`).

        Never raises: a transport error or a 4xx/5xx yields ``("", "")`` and the
        caller retries. The logs endpoint is the source of truth for the result,
        so one bad read must not fail the run.
        """
        try:
            resp = await client.get(self._logs_url(workload_id), headers=self._headers())
        except Exception:  # pragma: no cover — defensive
            logger.exception("workload-api logs fetch raised; will retry / fall back to logTail")
            return "", ""
        if resp.status_code >= 400:
            logger.warning(
                "workload-api logs fetch failed: %s %s",
                resp.status_code,
                resp.text,
            )
            return "", ""
        try:
            data = resp.json().get("data") or []
            return self._partition_log_entries(data)
        except Exception:  # pragma: no cover — defensive
            # A 200 that is not the documented JSON shape must not fail the run:
            # this endpoint is the source of truth for the result, so a bad read
            # is retried like a transport error. Kept inside the same contract
            # as the fetch above rather than escaping to _watch_logs.
            logger.exception(
                "workload-api logs response was not the expected JSON shape; "
                "will retry / fall back to logTail"
            )
            return "", ""

    @staticmethod
    def _marker_deadline(deadline: float, watch: _StatusWatch) -> float:
        """How long to keep waiting for the result marker, given what status knows.

        Reproduces the budgets the terminal-gated implementation used, with the
        terminal record now arriving mid-wait instead of before it:

        - **Status not terminal yet** → the overall run ``deadline``. The
          container may not even have started (image pull), which is exactly
          what the provisioning allowance exists for.
        - **Terminal FAILURE status** → ``max(deadline, terminal + flush)``. A
          one-shot's normal exit is flagged ``errored``, and a transient
          ``ErrImagePull`` errors the workload BEFORE the runner starts, with
          the marker arriving a minute later (#638).
        - **Any other terminal status** → ``terminal + flush``. The run is over;
          all that is left is collector latency.
        """
        if watch.terminal is None or watch.terminal_at is None:
            return deadline
        flush_deadline = watch.terminal_at + _LOG_FLUSH_TIMEOUT_S
        if watch.status in _TERMINAL_FAILURE:
            return max(deadline, flush_deadline)
        return flush_deadline

    async def _watch_logs(
        self,
        client: httpx.AsyncClient,
        workload_id: str,
        deadline: float,
        watch: _StatusWatch,
    ) -> tuple[str, str]:
        """Poll the logs endpoint until the result marker is available.

        Returns ``(stdout, stderr)``, preferring whichever source carries the
        marker: OTEL stdout first, then ``statusDetails.logTail`` off the
        terminal record once ``watch`` has one (logTail carries no per-line
        severity, so it is treated as stdout).

        Runs concurrently with the status poll, so it starts at submit rather
        than at terminal status — that is the marker-first latency win. Giving
        up therefore needs *both* halves of the old terminal-gated rule:
        non-empty-but-still output ends the wait after
        :data:`_STABLE_OUTPUT_NONEMPTY_QUIET_S` **only once ``watch`` has a
        terminal record**, because before that, still output means the container
        is running quietly rather than finished without a marker. Every other
        case (empty output, or still output pre-terminal) polls to
        :meth:`_marker_deadline`.
        """
        stdout = ""
        stderr = ""
        tail = ""
        prev_stdout: str | None = None
        stdout_settled_at = time.monotonic()
        delay = _LOG_POLL_INITIAL_DELAY_S
        while True:
            stdout, stderr = await self._read_logs_once(client, workload_id)
            # logTail only exists once the status poll has a terminal record; it
            # can carry the marker before — or instead of — the OTEL flush.
            if watch.terminal is not None:
                tail = self._extract_log_tail(watch.terminal)

            if has_result_marker(stdout) or has_result_marker(tail):
                break
            # A terminal `timeout` status raises SandboxTimeout in run()
            # regardless of what the logs say, so there is nothing left to wait
            # for. Break with the records already in hand rather than burning
            # the flush budget on an answer that cannot change.
            if watch.status in _TERMINAL_TIMEOUT:
                break
            if time.monotonic() > self._marker_deadline(deadline, watch):
                break
            # No marker yet. Decide whether one is still coming.
            #
            # Keyed on whether the container produced ANY output, not on how
            # long output has been still. Output means the runner is executing,
            # and it prints the marker after its own finally — so a marker is in
            # flight and the only question is collector latency, which we cannot
            # bound from here. Waiting on stillness alone is what made a
            # succeeded run report "no result marker": stdout sat unchanged at
            # the runner's first line while the marker was still being flushed,
            # and ERROR-level records (which route to stderr) kept arriving, so
            # the stream was demonstrably alive while `stdout` looked frozen.
            now = time.monotonic()
            if stdout != prev_stdout:
                prev_stdout = stdout
                stdout_settled_at = now
            elif (
                stdout
                and watch.terminal is not None
                and now - stdout_settled_at >= _STABLE_OUTPUT_NONEMPTY_QUIET_S
            ):
                # Non-empty, still, AND the workload has reached a terminal
                # status: the container can no longer produce output, so the
                # marker is not coming.
                #
                # The terminal condition is load-bearing now that this watcher
                # starts at submit instead of after the status poll. The runner
                # prints a startup line ("sandbox process limit ...") BEFORE
                # user code, so stdout goes non-empty immediately and then sits
                # unchanged for as long as the snippet runs quietly. Without the
                # terminal gate, a snippet that prints nothing for
                # _STABLE_OUTPUT_NONEMPTY_QUIET_S would be abandoned while its
                # container was still working, and `timeout_s` would stop
                # bounding user code the moment that first line appeared.
                # Before terminal status, still output means "running", not
                # "finished without a marker"; the wait is bounded by
                # _marker_deadline instead.
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, _LOG_POLL_MAX_DELAY_S)

        # Return whichever source carries the marker so the caller's
        # parse_result_marker finds it; otherwise prefer non-empty OTEL, then
        # fall back to logTail.
        if has_result_marker(stdout):
            return stdout, stderr
        if has_result_marker(tail):
            return (f"{stdout}\n{tail}" if stdout.strip() else tail), stderr
        if stdout.strip() or stderr.strip():
            return stdout, stderr
        return tail, stderr

    async def _watch_status(
        self,
        client: httpx.AsyncClient,
        workload_id: str,
        deadline: float,
        watch: _StatusWatch,
    ) -> None:
        """Run :meth:`_poll` and record its outcome on ``watch``.

        Swallows the poll's exceptions rather than propagating them, because
        status is no longer the gate: a result marker outranks a status-fetch
        failure or a poll deadline. :meth:`run` re-raises ``watch.error`` only
        when no marker was found either, so a genuine never-starts workload
        still fails with the poll's own error.
        """
        try:
            terminal = await self._poll(client, workload_id, deadline)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 — recorded, re-raised by run()
            watch.error = exc
            return
        watch.terminal = terminal
        watch.terminal_at = time.monotonic()

    async def _await_result(
        self,
        client: httpx.AsyncClient,
        workload_id: str,
        deadline: float,
    ) -> tuple[str, str, _StatusWatch]:
        """Watch container logs and workload status CONCURRENTLY (MODEL-24537).

        Returns ``(stdout, stderr, watch)`` as soon as the result marker is
        available, without waiting for Kubernetes to give up on a container that
        already ran (see the module docstring). ``watch`` carries whatever the
        status poll learned — possibly nothing, on a marker-first return.

        The status task is always cancelled and drained on the way out, so
        neither a marker-first return, an exception, nor caller cancellation
        leaves it pending.
        """
        watch = _StatusWatch()
        status_task = asyncio.create_task(
            self._watch_status(client, workload_id, deadline, watch),
            name=f"dr-sandbox-status-{workload_id}",
        )
        try:
            stdout, stderr = await self._watch_logs(client, workload_id, deadline, watch)
            stdout, stderr = await self._confirm_not_a_timeout(
                client, workload_id, stdout, stderr, watch
            )
        finally:
            status_task.cancel()
            # Drain it so nothing is left pending. _watch_status swallows every
            # non-cancellation exception, so the only thing awaiting can raise
            # here is the CancelledError we just requested — and if an OUTER
            # cancellation interrupts this await, the task is already cancelled
            # and finishes on its own without an unretrieved exception.
            with contextlib.suppress(asyncio.CancelledError):
                await status_task
        return stdout, stderr, watch

    async def _confirm_not_a_timeout(
        self,
        client: httpx.AsyncClient,
        workload_id: str,
        stdout: str,
        stderr: str,
        watch: _StatusWatch,
    ) -> tuple[str, str]:
        """Re-read the logs once when a null marker could be hiding a timeout.

        The runner's timeout path emits ``__DR_SANDBOX_RESULT__:null`` — the
        same bytes as a snippet that returned nothing — and identifies itself
        only by the stderr line it prints immediately beforehand
        (:func:`has_timeout_marker`). On a marker-first return there is no
        terminal record to check ``exitCode`` against, so that line is the whole
        signal, and it must not be missed to a collector batch boundary.

        Applies to the ambiguous shape ONLY: marker present, payload null, no
        sentinel seen yet, no terminal record. Everything else returns
        untouched, so the cost is bounded to :data:`_NULL_MARKER_RECHECK_S` on
        null-returning snippets and zero elsewhere.
        """
        if watch.terminal is not None:
            return stdout, stderr  # terminal record is authoritative
        if not has_result_marker(stdout):
            return stdout, stderr
        if has_timeout_marker(f"{stdout}\n{stderr}"):
            return stdout, stderr  # already decided — run() will raise
        if parse_result_marker(stdout)[1] is not None:
            return stdout, stderr  # a real return value is never the timeout path
        await asyncio.sleep(_NULL_MARKER_RECHECK_S)
        rechecked_stdout, rechecked_stderr = await self._read_logs_once(client, workload_id)
        # Only trust the re-read if it still has the marker; a transient bad
        # read returns ("", "") and must not erase what we already had.
        if has_result_marker(rechecked_stdout):
            return rechecked_stdout, rechecked_stderr
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
        # timeout_s bounds user code and is enforced in-container (SIGALRM →
        # exit 124); the poll deadline additionally allows for scheduling and
        # image pulls so infra time doesn't consume the user-code budget.
        deadline = start + timeout_s + self.provisioning_timeout_s

        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s + 5))
        workload_id: str | None = None
        try:
            workload_id = await self._submit(client, payload)
            # Logs and status are watched CONCURRENTLY and this returns at the
            # result marker, not at terminal status (MODEL-24537 — see the module
            # docstring for why terminal status is 15-20s of pure waiting).
            stdout_raw, stderr, watch = await self._await_result(client, workload_id, deadline)
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
        status = watch.status
        # ``exitCode`` only exists on a terminal record, which a marker-first
        # return does not have — and which, measured on staging, reports null
        # for every sandbox workload anyway (the record describes a long-running
        # *service*, not a job). So this was already always 0 here; see the
        # module docstring.
        exit_code = int((watch.terminal or {}).get("exitCode", 0) or 0)
        # The runner is a one-shot job, but the workload-api only supports
        # long-running "service" (or "nim") artifacts — so when the runner
        # finishes and its process exits, the workload-api marks the workload
        # ``errored``/``stopped`` even though the run succeeded. The runner emits
        # its result marker from a ``finally`` (see protocol.py / the sandbox
        # image runner), so a marker in the logs is the source of truth that the
        # snippet ran to completion. Treat failure statuses as real failures only
        # when no marker was produced (genuine startup/crash before any output).
        marker_found = has_result_marker(stdout_raw)

        if not marker_found and watch.terminal is None:
            # Nothing to go on from either source. The status poll's own error —
            # its deadline, or a status-fetch failure — is the real one. (When a
            # marker WAS found we deliberately ignore that error: the marker
            # proves the snippet ran, which outranks a status-endpoint problem.)
            raise watch.error or SandboxTimeout(
                f"sandbox exceeded timeout waiting for a result from workload {workload_id}"
            )

        if status in _TERMINAL_TIMEOUT:
            raise SandboxTimeout(
                f"workload-api workload {workload_id} timed out: {watch.terminal!r}",
                exit_code=exit_code,
                stderr=stderr,
            )
        # The runner self-kills on its in-process SIGALRM cap. Surface that as
        # SandboxTimeout so callers see one unified timeout path regardless of
        # which layer tripped first (datarobot/datarobot-user-models#2137).
        #
        # Two signals, because neither covers every case:
        #   * exit 124 on the terminal record — the original check. Kept for
        #     backends / API versions that populate exitCode, but measured on
        #     staging the sandbox workload record never does, so this cannot be
        #     relied on and never fires there.
        #   * the runner's own "sandbox exceeded timeout of Ns" line, which it
        #     prints before emitting its (null) result marker. This is in-band,
        #     so it works on a marker-first return where there is no terminal
        #     record at all. Checked across stdout AND stderr: measured on
        #     staging, the OTEL endpoint tags the runner's own writes to fd 2 as
        #     INFO (only the platform's own lrs-* records come through at ERROR
        #     severity), so the line arrives on the stdout side.
        if exit_code == SANDBOX_TIMEOUT_EXIT_CODE or has_timeout_marker(f"{stdout_raw}\n{stderr}"):
            raise SandboxTimeout(
                f"workload-api workload {workload_id} runner exceeded its "
                f"in-process timeout (exit {SANDBOX_TIMEOUT_EXIT_CODE}); "
                f"caller timeout_s={timeout_s}",
                exit_code=SANDBOX_TIMEOUT_EXIT_CODE,
                stderr=stderr,
            )
        if status in _TERMINAL_FAILURE and not marker_found:
            raise SandboxError(
                f"workload-api workload {workload_id} failed: status={status} "
                "(no result marker in logs — container did not run to completion)",
                exit_code=exit_code,
                stderr=stderr,
            )

        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            return_value=return_value,
            duration_s=duration,
            exit_code=exit_code,
        )
