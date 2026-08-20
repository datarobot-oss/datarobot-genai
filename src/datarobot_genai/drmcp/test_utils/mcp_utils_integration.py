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

import asyncio
import contextlib
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters
from mcp.client.stdio import stdio_client

from datarobot_genai.drmcp.test_utils.stub_credentials import STUB_DATAROBOT_API_TOKEN
from datarobot_genai.drmcp.test_utils.stub_credentials import STUB_DATAROBOT_ENDPOINT

# Default tool groups enabled for integration/acceptance MCP servers (model default is off).
_TEST_SUITE_ENABLED_TOOLS_ENV: dict[str, str] = {
    "ENABLE_PREDICTIVE_TOOLS": "true",
    "ENABLE_WORKLOAD_TOOLS": "true",
}


def integration_test_mcp_server_params(use_stub: bool = True) -> StdioServerParameters:
    # When running with stubs, always use stub credentials so the subprocess never attempts
    # a real dr.Client() version check against a staging/production endpoint from .env.
    api_token = STUB_DATAROBOT_API_TOKEN if use_stub else os.environ["DATAROBOT_API_TOKEN"]
    api_endpoint = STUB_DATAROBOT_ENDPOINT if use_stub else os.environ["DATAROBOT_ENDPOINT"]
    env = {
        "DATAROBOT_API_TOKEN": api_token,
        "DATAROBOT_ENDPOINT": api_endpoint,
        "MCP_SERVER_LOG_LEVEL": os.environ.get("MCP_SERVER_LOG_LEVEL") or "WARNING",
        "APP_LOG_LEVEL": os.environ.get("APP_LOG_LEVEL") or "WARNING",
        # Disable all OTEL telemetry for integration tests
        "OTEL_ENABLED": "false",
        "OTEL_SDK_DISABLED": "true",
        "OTEL_TRACES_EXPORTER": "none",
        "OTEL_LOGS_EXPORTER": "none",
        "OTEL_METRICS_EXPORTER": "none",
        "MCP_SERVER_REGISTER_DYNAMIC_TOOLS_ON_STARTUP": os.environ.get(
            "MCP_SERVER_REGISTER_DYNAMIC_TOOLS_ON_STARTUP"
        )
        or "false",
        "MCP_SERVER_REGISTER_DYNAMIC_PROMPTS_ON_STARTUP": os.environ.get(
            "MCP_SERVER_REGISTER_DYNAMIC_PROMPTS_ON_STARTUP"
        )
        or "true",
        **_TEST_SUITE_ENABLED_TOOLS_ENV,
    }

    script_dir = Path(__file__).resolve().parent
    server_script = str(script_dir / "integration_mcp_server.py")
    # Add src/ directory to Python path so datarobot_genai can be imported
    src_dir = script_dir.parent.parent.parent
    stub_flag = str(use_stub).lower()
    os.environ["MCP_USE_CLIENT_STUBS"] = stub_flag

    return StdioServerParameters(
        command="uv",
        args=["run", server_script],
        env={
            "PYTHONPATH": str(src_dir),
            "MCP_SERVER_NAME": "integration",
            "MCP_SERVER_PORT": "8081",
            "MCP_USE_CLIENT_STUBS": stub_flag,
            **env,
        },
    )


def integration_test_server_params_with_env(
    extra_env: dict[str, str],
    use_stub: bool = True,
) -> StdioServerParameters:
    """Return integration test server params with additional environment variables.

    Useful for enabling specific tool groups (e.g. {"ENABLE_USE_CASE_TOOLS": "true"}).
    """
    params = integration_test_mcp_server_params(use_stub=use_stub)
    env = dict(params.env or {})
    env.update(extra_env)
    return StdioServerParameters(
        command=params.command,
        args=params.args,
        env=env,
    )


class _SharedStdioSession:
    """A stdio MCP server subprocess + client session kept alive for reuse across tests.

    The stdio_client/ClientSession context managers are entered and exited inside a
    single dedicated task: their anyio cancel scopes must not cross task boundaries,
    which is what would happen if a session-scoped fixture entered them in one test's
    task and exited them in another's.
    """

    def __init__(self, server_params: StdioServerParameters, timeout: int) -> None:
        self._server_params = server_params
        self._timeout = timeout
        self.session: ClientSession | None = None
        self._ready = asyncio.Event()
        self._closing = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._startup_error: BaseException | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._serve())
        await self._ready.wait()
        if self._startup_error is not None:
            raise self._startup_error

    async def _serve(self) -> None:
        try:
            async with stdio_client(self._server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    init_result = await asyncio.wait_for(
                        session.initialize(), timeout=self._timeout
                    )
                    # Store the init result for tests that need to inspect capabilities
                    session._init_result = init_result  # type: ignore[attr-defined]
                    self.session = session
                    self._ready.set()
                    await self._closing.wait()
        except TimeoutError:
            self._startup_error = TimeoutError(
                f"Session initialization timed out after {self._timeout} seconds"
            )
        except BaseException as exc:
            self._startup_error = exc
        finally:
            self._ready.set()

    async def aclose(self) -> None:
        self._closing.set()
        if self._task is not None:
            with contextlib.suppress(BaseException):
                await self._task


# Shared sessions keyed by (event loop, command, args, env) so every test that asks for
# the same server configuration on the same loop reuses one subprocess instead of
# paying the ~4s server startup per test.
_shared_sessions: dict[tuple[Any, ...], _SharedStdioSession] = {}


def _shared_session_key(server_params: StdioServerParameters) -> tuple[Any, ...]:
    env = server_params.env or {}
    return (
        asyncio.get_running_loop(),
        server_params.command,
        tuple(server_params.args),
        tuple(sorted(env.items())),
    )


async def _get_shared_stdio_session(
    server_params: StdioServerParameters, timeout: int
) -> _SharedStdioSession:
    key = _shared_session_key(server_params)
    holder = _shared_sessions.get(key)
    if holder is None:
        holder = _SharedStdioSession(server_params, timeout)
        await holder.start()
        _shared_sessions[key] = holder
    return holder


async def aclose_shared_stdio_sessions() -> None:
    """Close every shared session bound to the current event loop."""
    loop = asyncio.get_running_loop()
    for key in [k for k in _shared_sessions if k[0] is loop]:
        await _shared_sessions.pop(key).aclose()


@contextlib.asynccontextmanager
async def integration_test_mcp_session(
    server_params: StdioServerParameters | None = None,
    timeout: int = 60,
    elicitation_callback: Any | None = None,
    use_stub: bool = True,
    shared: bool = True,
) -> AsyncGenerator[ClientSession, None]:
    """
    Create and connect a client for the MCP server as a context manager.

    By default the underlying stdio server subprocess and client session are shared
    across tests with the same server configuration (see _SharedStdioSession); exiting
    the context manager does not close them. Callers that need session-level isolation
    (a custom elicitation_callback, or tests that mutate server state) get a fresh
    server: pass ``shared=False``, or any elicitation_callback implies it.

    Args:
        server_params: Parameters for configuring the server connection
        timeout: Timeout
        elicitation_callback: Optional callback for handling elicitation requests
        shared: Reuse one server/session per configuration (default). False forces a
            fresh, test-scoped server subprocess.

    Yields
    ------
        ClientSession: Connected MCP client session

    Raises
    ------
        ConnectionError: If session initialization fails
        TimeoutError: If session initialization exceeds timeout
    """
    server_params = server_params or integration_test_mcp_server_params(use_stub=use_stub)

    if shared and elicitation_callback is None:
        holder = await _get_shared_stdio_session(server_params, timeout)
        assert holder.session is not None
        yield holder.session
        return

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(
                read_stream, write_stream, elicitation_callback=elicitation_callback
            ) as session:
                init_result = await asyncio.wait_for(session.initialize(), timeout=timeout)
                # Store the init result on the session for tests that need to inspect capabilities
                session._init_result = init_result  # type: ignore[attr-defined]
                yield session

    except TimeoutError:
        raise TimeoutError(f"Session initialization timed out after {timeout} seconds")
