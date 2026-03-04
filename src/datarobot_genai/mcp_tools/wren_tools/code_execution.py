"""Code execution tool with pluggable sandbox backend.

The tool itself is generic — it accepts code and returns results.
The sandbox provider determines HOW the code runs (in-process,
sidecar container, serverless function).
"""
from __future__ import annotations

import asyncio
import logging
import traceback
from io import StringIO
from typing import Any, Protocol

from datarobot_genai.mcp_tools._registry import register_tool

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
        result: Any = None
        error: str | None = None

        import sys

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        stderr_buf = StringIO()

        namespace: dict[str, Any] = {}

        def _run_sync() -> None:
            nonlocal result, error
            sys.stdout = stdout_buf
            sys.stderr = stderr_buf
            try:
                exec(compile(code, "<mcp_tool>", "exec"), namespace)  # noqa: S102
                result = namespace.get("result")
            except Exception:
                error = traceback.format_exc()
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, _run_sync),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            error = f"Execution timed out after {timeout_seconds} seconds"

        return {
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "result": result,
            "error": error,
        }


class NoopSandbox:
    """Placeholder for Global MCP until sidecar/serverless is built.

    Returns an error explaining that sandboxed execution is not yet
    available in this environment.
    """

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


# The tool function — sandbox is injected at server startup
_sandbox: SandboxProvider = NoopSandbox()


def set_sandbox_provider(provider: SandboxProvider) -> None:
    global _sandbox
    _sandbox = provider


async def execute_code(
    code: str,
    session_id: str | None = None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Execute Python code in a sandboxed environment.

    The execution environment depends on the server configuration:
    - Template MCP (single-tenant): runs in-process with restricted timeout
    - Global MCP (multi-tenant): returns an error (sandboxed execution pending)

    Returns a dict with keys: stdout, stderr, result, error.
    """
    return await _sandbox.execute(code, session_id or "default", timeout_seconds)


register_tool(
    name="execute_code",
    func=execute_code,
    description=(
        "Execute Python code in a sandboxed environment. "
        "Returns stdout, stderr, and optional result value."
    ),
    category="wren_tools",
)
