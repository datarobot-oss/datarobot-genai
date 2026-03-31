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
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.sessions import SSEConnection
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import PrivateAttr

from datarobot_genai.core.mcp.config import MCPConfig


def _wrap_mcp_tool_for_langgraph(inner: BaseTool) -> BaseTool:
    """Wrap an MCP tool so LangGraph-injected 'runtime' is filtered from callback inputs.

    MCP tools from langchain_mcp_adapters use args_schema=tool.inputSchema (a dict).
    LangChain's _filter_injected_args only filters keys declared on Pydantic args_schema,
    so it never filters 'runtime'. That leaves ToolRuntime (with config/callbacks) in the
    inputs passed to profiler callbacks, which then fail on copy.deepcopy (e.g. coroutines).

    This wrapper declares 'runtime' as an injected arg so it is filtered before callbacks
    see the inputs, while delegating execution to the inner tool unchanged.
    """

    async def _invoke_inner(**kwargs: Any) -> Any:
        # Call the inner tool's coroutine so we return (content, artifact), not the
        # formatted output that ainvoke() would return.
        if getattr(inner, "coroutine", None) is not None:
            return await inner.coroutine(**kwargs)
        return await inner.ainvoke(kwargs)

    class _MCPToolWrapper(StructuredTool):
        """Thin wrapper that adds 'runtime' to injected args for callback filtering."""

        _inner: BaseTool = PrivateAttr()

        def __init__(self, inner_tool: BaseTool, coro: Any) -> None:
            super().__init__(
                name=inner_tool.name,
                description=inner_tool.description or "",
                args_schema=inner_tool.args_schema,
                coroutine=coro,
                response_format=getattr(inner_tool, "response_format", "content_and_artifact"),
                metadata=getattr(inner_tool, "metadata", None),
            )
            self._inner = inner_tool

        @property
        def _injected_args_keys(self) -> frozenset[str]:
            base: frozenset[str] = getattr(self._inner, "_injected_args_keys", frozenset())
            return base | frozenset(["runtime"])

    return _MCPToolWrapper(inner, _invoke_inner)


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
        raw_tools = await load_mcp_tools(session=session)
        # Wrap so LangGraph-injected 'runtime' is filtered from callback inputs (avoids
        # copy.deepcopy of non-serializable ToolRuntime in NAT profiler).
        tools = [_wrap_mcp_tool_for_langgraph(t) for t in raw_tools]
        print(f"Successfully loaded {len(tools)} MCP tools", flush=True)
        yield tools
