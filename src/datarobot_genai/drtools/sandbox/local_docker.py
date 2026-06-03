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

"""Local-Docker backed sandbox for development and testing.

Runs the *same* sandbox image as :class:`DataRobotWorkloadSandbox`, with the
*same* runner protocol (``DR_SANDBOX_CODE_B64`` / ``DR_SANDBOX_INPUTS_B64`` /
``DR_SANDBOX_TIMEOUT_SECS`` env vars and the ``__DR_SANDBOX_RESULT__:`` stdout
marker), but launches it as a local container via the ``docker`` CLI instead
of submitting a workload to the DataRobot workload-api. This makes it a
high-fidelity dev/test counterpart — real CPython, real libraries, the real
runner — that needs only a local Docker daemon, no DataRobot endpoint.

.. note::
   Isolation is whatever the local container + the applied security flags
   provide. For untrusted code in production, use the workload-api backend.
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any

from datarobot_genai.drtools.sandbox.base import SandboxError
from datarobot_genai.drtools.sandbox.base import SandboxResult
from datarobot_genai.drtools.sandbox.base import SandboxSecurityContext
from datarobot_genai.drtools.sandbox.base import SandboxTimeout
from datarobot_genai.drtools.sandbox.protocol import SANDBOX_TIMEOUT_EXIT_CODE
from datarobot_genai.drtools.sandbox.protocol import parse_result_marker

logger = logging.getLogger(__name__)

# Extra wall-clock beyond the caller's timeout_s before we give up on the
# `docker run` client and force-kill it — covers image pull / container
# startup so a slow cold start isn't misreported as a user-code timeout.
_CLIENT_GRACE_S = 15.0

# Bounded wait for the best-effort `docker rm -f` teardown.
_TEARDOWN_TIMEOUT_S = 5.0


class LocalDockerSandbox:
    """Sandbox that runs the sandbox image in a local Docker container.

    Parameters
    ----------
    image
        Container image URI to run. Should be the same DRUM-built sandbox
        image used by :class:`DataRobotWorkloadSandbox` so behavior matches
        production.
    docker_bin
        Name/path of the Docker CLI binary. Defaults to ``"docker"``.
    security_context
        Optional :class:`SandboxSecurityContext`. When provided, its
        :meth:`~SandboxSecurityContext.to_docker_run_args` flags are passed to
        ``docker run`` (read-only rootfs, drop caps, no-new-privileges, …).
        When ``None`` (the default) no hardening flags are added beyond
        ``network_disabled`` / ``memory_limit``.
    network_disabled
        Run with ``--network none`` (default ``True``) — sandboxed code has no
        network access.
    memory_limit
        Value for ``docker run --memory`` (default ``"512m"``, matching the
        workload-api ``resourceRequest``). Pass ``None`` to omit.
    extra_run_args
        Additional raw ``docker run`` arguments appended verbatim (escape
        hatch for mounts, env, etc.).
    """

    def __init__(
        self,
        image: str,
        *,
        docker_bin: str = "docker",
        security_context: SandboxSecurityContext | None = None,
        network_disabled: bool = True,
        memory_limit: str | None = "512m",
        extra_run_args: list[str] | None = None,
    ) -> None:
        self.image = image
        self.docker_bin = docker_bin
        self.security_context = security_context
        self.network_disabled = network_disabled
        self.memory_limit = memory_limit
        self.extra_run_args = list(extra_run_args or [])

    def _run_args(
        self, container_name: str, code: str, inputs: dict[str, Any] | None, timeout_s: float
    ) -> list[str]:
        code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
        inputs_b64 = base64.b64encode(json.dumps(inputs or {}).encode("utf-8")).decode("ascii")
        # Floor of 1s so the runner doesn't read 0 (which disables its own cap).
        runner_timeout = max(1, int(timeout_s))
        args = [self.docker_bin, "run", "--rm", "--name", container_name]
        if self.network_disabled:
            args += ["--network", "none"]
        if self.memory_limit:
            args += ["--memory", self.memory_limit]
        if self.security_context is not None:
            args += self.security_context.to_docker_run_args()
        args += self.extra_run_args
        args += [
            "-e",
            f"DR_SANDBOX_CODE_B64={code_b64}",
            "-e",
            f"DR_SANDBOX_INPUTS_B64={inputs_b64}",
            "-e",
            f"DR_SANDBOX_TIMEOUT_SECS={runner_timeout}",
            self.image,
        ]
        return args

    async def _force_remove(self, container_name: str) -> None:
        """Best-effort ``docker rm -f``; swallow all errors and never raise."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.docker_bin,
                "rm",
                "-f",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=_TEARDOWN_TIMEOUT_S)
        except Exception:  # noqa: BLE001 — teardown must never mask the real result
            logger.debug("docker rm -f %s failed; ignoring", container_name)

    async def run(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        externals: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> SandboxResult:
        """Execute ``code`` in a local Docker container. See :class:`Sandbox.run`."""
        if externals:
            raise NotImplementedError(
                "externals injection is not supported by LocalDockerSandbox; "
                "the container receives only base64 code + inputs over env vars."
            )

        container_name = f"dr-sandbox-{uuid.uuid4().hex[:12]}"
        args = self._run_args(container_name, code, inputs, timeout_s)
        start = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise SandboxError(
                f"docker binary {self.docker_bin!r} not found; install Docker or "
                "pass docker_bin / use DataRobotWorkloadSandbox."
            ) from exc

        try:
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s + _CLIENT_GRACE_S
                )
            except TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise SandboxTimeout(
                    f"local docker sandbox exceeded timeout_s={timeout_s}"
                ) from exc
        finally:
            # Force teardown on success, failure, timeout, or cancellation —
            # `--rm` won't fire if we had to kill the client.
            await self._force_remove(container_name)

        duration = time.monotonic() - start
        exit_code = proc.returncode if proc.returncode is not None else 0
        stdout_raw = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        stdout, return_value = parse_result_marker(stdout_raw)

        # The runner exits SANDBOX_TIMEOUT_EXIT_CODE when its in-process cap
        # fires first; surface as SandboxTimeout for one unified timeout path.
        if exit_code == SANDBOX_TIMEOUT_EXIT_CODE:
            raise SandboxTimeout(
                f"local docker sandbox runner exceeded its in-process timeout "
                f"(exit {SANDBOX_TIMEOUT_EXIT_CODE}); caller timeout_s={timeout_s}"
            )
        if exit_code != 0:
            detail = stderr.strip() or stdout.strip() or "no output"
            raise SandboxError(f"local docker sandbox exited {exit_code}: {detail}")

        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            return_value=return_value,
            duration_s=duration,
            exit_code=exit_code,
        )
