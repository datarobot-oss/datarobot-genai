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

from .utils import load_env

load_env()


def integration_test_mcp_server_params() -> StdioServerParameters:
    env = {
        "DATAROBOT_API_TOKEN": os.environ.get("DATAROBOT_API_TOKEN") or "test-token",
        "DATAROBOT_ENDPOINT": os.environ.get("DATAROBOT_ENDPOINT")
        or "https://test.datarobot.com/api/v2",
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
    }

    script_dir = Path(__file__).resolve().parent
    server_script = str(script_dir / "integration_mcp_server.py")
    # Add src/ directory to Python path so datarobot_genai can be imported
    src_dir = script_dir.parent.parent.parent

    return StdioServerParameters(
        command="uv",
        args=["run", server_script],
        env={
            "PYTHONPATH": str(src_dir),
            "MCP_SERVER_NAME": "integration",
            "MCP_SERVER_PORT": "8081",
            **env,
        },
    )


@contextlib.asynccontextmanager
async def integration_test_mcp_session(
    server_params: StdioServerParameters | None = None,
    timeout: int = 30,
    elicitation_callback: Any | None = None,
) -> AsyncGenerator[ClientSession, None]:
    """
    Create and connect a client for the MCP server as a context manager.

    Args:
        server_params: Parameters for configuring the server connection
        timeout: Timeout
        elicitation_callback: Optional callback for handling elicitation requests

    Yields
    ------
        ClientSession: Connected MCP client session

    Raises
    ------
        ConnectionError: If session initialization fails
        TimeoutError: If session initialization exceeds timeout
    """
    server_params = server_params or integration_test_mcp_server_params()

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(
                read_stream, write_stream, elicitation_callback=elicitation_callback
            ) as session:
                init_result = await asyncio.wait_for(session.initialize(), timeout=timeout)
                # Store the init result on the session for tests that need to inspect capabilities
                session._init_result = init_result  # type: ignore[attr-defined]
                yield session

    except asyncio.TimeoutError:
        raise TimeoutError(f"Session initialization timed out after {timeout} seconds")
