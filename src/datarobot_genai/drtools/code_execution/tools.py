# Copyright 2025 DataRobot, Inc.
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

"""Code execution tool with pluggable sandbox backend."""

import asyncio
import contextlib
import logging
import traceback
from io import StringIO
from typing import Annotated
from typing import Any
from typing import Protocol

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp import dr_mcp_integration_tool

logger = logging.getLogger(__name__)


class SandboxProvider(Protocol):
    async def execute(
        self, code: str, session_id: str, timeout_seconds: int = 30
    ) -> dict[str, Any]:
        """Execute code in isolated environment. Returns stdout, stderr, result."""
        ...


class InProcessSandbox:
    """For single-tenant template MCP — runs code in-process.

    WARNING: Only safe for single-tenant deployments where the customer
    controls their own environment.
    """

    async def execute(
        self, code: str, session_id: str, timeout_seconds: int = 30
    ) -> dict[str, Any]:
        stdout_buf = StringIO()
        stderr_buf = StringIO()
        result: Any = None
        error: str | None = None

        namespace: dict[str, Any] = {}

        def _run_sync() -> None:
            nonlocal result, error
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                try:
                    exec(compile(code, "<mcp_tool>", "exec"), namespace)  # noqa: S102
                    result = namespace.get("result")
                except BaseException:
                    error = traceback.format_exc()

        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, _run_sync),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            error = f"Execution timed out after {timeout_seconds} seconds"

        return {
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "result": result,
            "error": error,
        }


class NoopSandbox:
    """Placeholder for Global MCP until sidecar/serverless is built."""

    async def execute(
        self, code: str, session_id: str, timeout_seconds: int = 30
    ) -> dict[str, Any]:
        return {
            "error": (
                "Code execution not available in this environment. "
                "Sandboxed execution is a planned future capability."
            ),
            "stdout": "",
            "stderr": "",
            "result": None,
        }


_sandbox: SandboxProvider = NoopSandbox()


def set_sandbox_provider(provider: SandboxProvider) -> None:
    global _sandbox  # noqa: PLW0603
    _sandbox = provider


@dr_mcp_integration_tool(tags={"code_execution", "python", "sandbox", "daria"})
async def execute_code(
    *,
    code: Annotated[str, "The Python code to execute"] | None = None,
    session_id: Annotated[str, "Optional session ID (reserved for future stateful execution)"]
    | None = None,
    timeout_seconds: Annotated[int, "Execution timeout in seconds"] = 30,
) -> ToolResult:
    """Execute Python code in a sandboxed environment.

    The execution environment depends on the server configuration:
    - Template MCP (single-tenant): runs in-process with restricted timeout
    - Global MCP (multi-tenant): returns an error (sandboxed execution pending)
    """
    if not code:
        raise ToolError("Code must be provided")

    result = await _sandbox.execute(code, session_id or "default", timeout_seconds)
    return ToolResult(structured_content=result)
