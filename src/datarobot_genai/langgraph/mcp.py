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

import copy
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from langchain.tools import BaseTool
from langchain_core.tools import StructuredTool
from langchain_core.tools.base import ToolException
from langchain_mcp_adapters.sessions import SSEConnection
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import PrivateAttr

from datarobot_genai.core.mcp import MCPConfig

logger = logging.getLogger(__name__)


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
        try:
            if getattr(inner, "coroutine", None) is not None:
                return await inner.coroutine(**kwargs)
            return await inner.ainvoke(kwargs)
        except ToolException as exc:
            logger.warning("MCP tool '%s' raised ToolException: %s", inner.name, exc)
            error_content = f"Tool '{inner.name}' failed: {exc}"
            response_format = getattr(inner, "response_format", "content_and_artifact")
            if response_format == "content_and_artifact":
                return error_content, None
            return error_content

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
    mcp_config: MCPConfig,
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
    server_config = mcp_config.server_config

    if not server_config:
        logger.info("No MCP server configured, using empty tools list")
        yield []
        return

    # Prevent mutation of the original server_config
    server_config = copy.deepcopy(server_config)

    url = server_config["url"]
    logger.info("Connecting to MCP server: %s", url)

    # Pop transport from server_config to avoid passing it twice
    # Use .pop() with default to never error
    transport = server_config.pop("transport", "streamable-http")

    if transport in ["streamable-http", "streamable_http"]:
        connection = StreamableHttpConnection(transport="streamable_http", **server_config)
    elif transport == "sse":
        connection = SSEConnection(transport="sse", **server_config)
    else:
        raise RuntimeError("Unsupported MCP transport specified.")

    # Graceful fallback: if we can't connect to the MCP server, yield empty tools
    # instead of crashing.
    #
    # The `connected` flag distinguishes two cases for the except clause:
    #   - Exception raised during setup (before yield): log a warning and yield [].
    #   - Exception thrown back in from the consumer (after yield, via athrow()): re-raise.
    # Without this guard, a consumer exception of a caught type would hit `yield []` as a
    # second yield, causing `RuntimeError: generator didn't stop after athrow()`.
    connected = False
    try:
        # Load tools WITHOUT holding a persistent MCP session open across the stream.
        # ``session=None`` + ``connection`` makes ``load_mcp_tools`` open a short-lived
        # session -- entered and exited within this await, in one task, before any
        # streaming yield -- to list the tools; each returned tool then opens its own
        # per-call session. Holding a persistent session across the streaming response
        # (the previous ``async with create_session(...)``) keeps the MCP streamable-http
        # client's anyio task group open for the whole run. On the HITL interrupt path the
        # response generator is finalized in a different task than it was entered, so that
        # task group's cancel scope is exited in the wrong task ("Attempted to exit cancel
        # scope in a different task than it was entered in"), which cancels the run and
        # drops the buffered interrupt events. Per-call sessions have no long-lived task
        # group spanning the stream, so the teardown stays task-local.
        raw_tools = await load_mcp_tools(session=None, connection=connection)
        tools = [_wrap_mcp_tool_for_langgraph(t) for t in raw_tools]
        logger.info("Successfully loaded %d MCP tools", len(tools))
        connected = True
        yield tools
    except (ConnectionError, OSError, TimeoutError, ExceptionGroup) as exc:
        if connected:
            raise
        logger.warning(
            "Failed to connect to MCP server at %s: %s. Continuing without MCP tools.",
            url,
            exc,
        )
        yield []
