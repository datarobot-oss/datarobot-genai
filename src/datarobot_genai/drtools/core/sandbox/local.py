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

"""Local, NON-SANDBOXED code execution backend.

:class:`LocalProcessSandbox` implements the :class:`Sandbox` protocol by
running the snippet in a plain subprocess of the current Python interpreter.
It exists solely for the ``MCP_SANDBOX_DISABLED`` kill-switch (see
:mod:`datarobot_genai.drmcputils.sandbox_mode`): local development, and
deployments where the DataRobot workload-api sandbox is not available.

SECURITY: this backend provides **no isolation**. The code runs with the MCP
server's user, filesystem, network, and environment (minus the handful of
variables this module sets). The only guardrails are a wall-clock timeout and
a separate process (so a crash cannot take down the server).

Why not Monty? FastMCP 3.4.x ships ``MontySandboxProvider`` backed by
``pydantic-monty``, a restricted Rust interpreter — but Monty cannot import
real Python packages (``import polars`` fails with ``ModuleNotFoundError``
even when polars is installed), and the panel transform/filter tools execute
polars-based snippets (see ``drtools/panels/transform.py``). A subprocess of
the host interpreter runs the exact same code the workload sandbox image
would, at the cost of the isolation the kill-switch already forgoes.

The child reuses the workload runner's wire contract
(:mod:`datarobot_genai.drtools.core.sandbox.protocol`): ``inputs`` is bound in
the executing namespace, the code may assign ``_return``, and the result is
emitted as a final ``__DR_SANDBOX_RESULT__:<json>`` stdout line.
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
from typing import Any

from datarobot_genai.drtools.core.sandbox.base import SandboxError
from datarobot_genai.drtools.core.sandbox.base import SandboxResult
from datarobot_genai.drtools.core.sandbox.base import SandboxTimeout
from datarobot_genai.drtools.core.sandbox.protocol import RESULT_MARKER
from datarobot_genai.drtools.core.sandbox.protocol import parse_result_marker

logger = logging.getLogger(__name__)

_CODE_ENV_VAR = "DR_SANDBOX_CODE_B64"
_INPUTS_ENV_VAR = "DR_SANDBOX_INPUTS_B64"
_MARKER_ENV_VAR = "DR_SANDBOX_RESULT_MARKER"

# Executed via ``python -c`` in the child. Reads code/inputs/marker from env
# vars (mirroring the workload runner's DR_SANDBOX_CODE_B64 / _INPUTS_B64
# contract) so no quoting/escaping of user code is ever needed. On user-code
# failure it prints the traceback to stderr and exits 1; on success it emits
# the result-marker line last so ``parse_result_marker`` finds it.
_CHILD_BOOTSTRAP = f"""
import base64, json, os, sys, traceback

code = base64.b64decode(os.environ.pop({_CODE_ENV_VAR!r})).decode("utf-8")
inputs = json.loads(base64.b64decode(os.environ.pop({_INPUTS_ENV_VAR!r})).decode("utf-8"))
marker = os.environ.pop({_MARKER_ENV_VAR!r})

namespace = {{"inputs": inputs, "__name__": "__main__"}}
try:
    exec(compile(code, "<dr-local-sandbox>", "exec"), namespace)
except BaseException:
    traceback.print_exc()
    sys.exit(1)

try:
    encoded = json.dumps(namespace.get("_return"), default=str)
except (TypeError, ValueError):
    encoded = "null"
sys.stdout.flush()
print(marker + encoded, flush=True)
"""


class LocalProcessSandbox:
    """``Sandbox`` implementation that executes code in a local subprocess.

    NOT a sandbox: no filesystem, network, or resource isolation — see the
    module docstring. Selected only via the ``MCP_SANDBOX_DISABLED``
    kill-switch.

    Parameters
    ----------
    python_executable
        Interpreter to run the snippet with. Defaults to ``sys.executable``,
        so the child sees the same installed packages (polars, etc.) as the
        MCP server.
    """

    def __init__(self, python_executable: str | None = None) -> None:
        self.python_executable = python_executable or sys.executable

    async def run(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        externals: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> SandboxResult:
        """Execute ``code`` in a local subprocess. See :class:`Sandbox.run`."""
        if externals:
            raise NotImplementedError(
                "externals injection is not supported by LocalProcessSandbox; "
                "it mirrors DataRobotWorkloadSandbox, where externals are a "
                "follow-up tied to CodeMode tool-catalog serialization."
            )

        env = dict(os.environ)
        env[_CODE_ENV_VAR] = base64.b64encode(code.encode("utf-8")).decode("ascii")
        env[_INPUTS_ENV_VAR] = base64.b64encode(json.dumps(inputs or {}).encode("utf-8")).decode(
            "ascii"
        )
        env[_MARKER_ENV_VAR] = RESULT_MARKER

        start = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            self.python_executable,
            "-c",
            _CHILD_BOOTSTRAP,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout_s
            )
        except TimeoutError as exc:
            process.kill()
            # Reap the killed child so no zombie is left behind.
            await process.communicate()
            raise SandboxTimeout(f"local execution exceeded timeout of {timeout_s}s") from exc
        except asyncio.CancelledError:
            process.kill()
            await process.communicate()
            raise

        duration = time.monotonic() - start
        stdout_raw = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        exit_code = process.returncode if process.returncode is not None else -1

        if exit_code != 0:
            tail = stderr.strip().splitlines()[-1] if stderr.strip() else ""
            raise SandboxError(
                f"local execution failed with exit code {exit_code}"
                + (f": {tail}" if tail else ""),
                exit_code=exit_code,
                stderr=stderr,
            )

        stdout, return_value = parse_result_marker(stdout_raw)
        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            return_value=return_value,
            duration_s=duration,
            exit_code=exit_code,
        )
