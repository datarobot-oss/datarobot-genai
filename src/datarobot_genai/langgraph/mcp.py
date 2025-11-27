# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from langchain.tools import BaseTool
from langchain_mcp_adapters.sessions import SSEConnection
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import load_mcp_tools

from datarobot_genai.core.mcp.common import MCPConfig


@asynccontextmanager
async def mcp_tools_context(
    authorization_context: dict[str, Any] | None = None,
    forwarded_headers: dict[str, str] | None = None,
) -> AsyncGenerator[list[BaseTool], None]:
    """Yield a list of LangChain BaseTool instances loaded via MCP.

    If no configuration or loading fails, yields an empty list without raising.

    Parameters
    ----------
    authorization_context : dict[str, Any] | None
        Authorization context to use for MCP connections
    forwarded_headers : dict[str, str] | None
        Forwarded headers, e.g. x-datarobot-api-key to use for MCP authentication
    """
    mcp_config = MCPConfig(
        authorization_context=authorization_context,
        forwarded_headers=forwarded_headers,
    )
    server_config = mcp_config.server_config

    if not server_config:
        print("No MCP server configured, using empty tools list", flush=True)
        yield []
        return

    url = server_config["url"]
    print(f"Connecting to MCP server: {url}", flush=True)

    # Pop transport from server_config to avoid passing it twice
    # Use .pop() with default to never error
    transport = server_config.pop("transport", "streamable-http")

    if transport in ["streamable-http", "streamable_http"]:
        connection = StreamableHttpConnection(transport="streamable_http", **server_config)
    elif transport == "sse":
        connection = SSEConnection(transport="sse", **server_config)
    else:
        raise RuntimeError("Unsupported MCP transport specified.")

    async with create_session(connection=connection) as session:
        # Use the connection to load available MCP tools
        tools = await load_mcp_tools(session=session)
        print(f"Successfully loaded {len(tools)} MCP tools", flush=True)
        yield tools
