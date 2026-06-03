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

"""Core types for the sandbox abstraction.

Defines the :class:`Sandbox` Protocol, the :class:`SandboxResult` and
:class:`SandboxSecurityContext` data models, and the exception hierarchy
(:class:`SandboxError`, :class:`SandboxTimeout`).
"""

from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import Protocol
from typing import runtime_checkable

from pydantic import BaseModel
from pydantic import Field


class SandboxError(Exception):
    """Base error raised when sandboxed execution fails.

    Subclassed by :class:`SandboxTimeout` for the specific timeout case.
    """


class SandboxTimeout(SandboxError):  # noqa: N818  # public API name; "Timeout" intentional
    """Raised when sandboxed execution exceeds the configured timeout."""


@dataclass
class SandboxResult:
    """Result of a successful sandboxed execution.

    Attributes
    ----------
    stdout
        Captured standard output.
    stderr
        Captured standard error.
    return_value
        Value the user code assigned to the magic ``_return`` variable, if any.
    duration_s
        Wall-clock execution duration in seconds.
    exit_code
        Process exit code (``0`` for success).
    """

    stdout: str
    stderr: str
    return_value: Any
    duration_s: float
    exit_code: int


SeccompProfileType = Literal["RuntimeDefault", "Localhost", "Unconfined"]


class SandboxSecurityContext(BaseModel):
    """Container security context applied to sandboxed workloads.

    Mirrors the ``SecurityContext`` schema in the DataRobot workload-api
    (see ``workload_api/schemas/containers.py:169``).

    Defaults are deliberately the most restrictive settings that a non-admin
    user is allowed to apply: read-only root filesystem, drop all Linux
    capabilities, ``RuntimeDefault`` seccomp profile, and no privilege
    escalation.
    """

    read_only_root_filesystem: bool = True
    capabilities_drop: list[str] = Field(default_factory=lambda: ["ALL"])
    capabilities_add: list[str] = Field(default_factory=list)
    seccomp_profile_type: SeccompProfileType = "RuntimeDefault"
    allow_privilege_escalation: bool = False

    def to_workload_api_dict(self) -> dict[str, Any]:
        """Serialize to the camelCase shape accepted by workload-api.

        Returns
        -------
        dict
            A dict with keys ``readOnlyRootFilesystem``,
            ``allowPrivilegeEscalation``, ``capabilities`` (with ``add``/``drop``
            sublists), and ``seccompProfile`` (with ``type``). Empty
            capability lists are omitted; ``capabilities`` and
            ``seccompProfile`` are omitted entirely when they would be empty.
        """
        payload: dict[str, Any] = {
            "readOnlyRootFilesystem": self.read_only_root_filesystem,
            "allowPrivilegeEscalation": self.allow_privilege_escalation,
        }

        capabilities: dict[str, list[str]] = {}
        if self.capabilities_drop:
            capabilities["drop"] = list(self.capabilities_drop)
        if self.capabilities_add:
            capabilities["add"] = list(self.capabilities_add)
        if capabilities:
            payload["capabilities"] = capabilities

        payload["seccompProfile"] = {"type": self.seccomp_profile_type}

        return payload


@runtime_checkable
class Sandbox(Protocol):
    """Protocol for sandboxed Python code execution.

    The production implementation is :class:`DataRobotWorkloadSandbox`
    (runs in the DataRobot workload-api). A local-Docker dev/test
    implementation is planned as a follow-up.
    """

    async def run(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        externals: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> SandboxResult:
        """Execute ``code`` in an isolated environment and return the result.

        Parameters
        ----------
        code
            Python source to execute. May assign to a magic ``_return``
            variable to communicate a return value back to the caller.
        inputs
            Mapping bound as ``inputs`` in the executing namespace. Must be
            JSON-serializable for remote backends.
        externals
            Mapping bound as ``externals`` in the executing namespace for
            CodeMode-style tool injection. Most implementations do not yet
            support this and will raise ``NotImplementedError``.
        timeout_s
            Wall-clock timeout in seconds.

        Returns
        -------
        SandboxResult
            stdout/stderr capture, the ``_return`` value, duration, and
            exit code.

        Raises
        ------
        SandboxTimeout
            When execution exceeds ``timeout_s``.
        SandboxError
            When the sandbox itself fails or the user code raises.
        """
        ...
