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

import logging
import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

from langchain.tools import BaseTool
from langchain_core.tools import StructuredTool
from langchain_core.tools.base import ToolException
from langchain_mcp_adapters.sessions import SSEConnection
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import PrivateAttr

from datarobot_genai.core.mcp.target import MCPTarget
from datarobot_genai.core.mcp.target import MCPTargetKind
from datarobot_genai.core.mcp.target import build_server_config

logger = logging.getLogger(__name__)


def _local_server_reachable(url: str, timeout: float = 1.0) -> bool:
    """TCP-probe a local MCP server's host:port.

    A local server that has not been started otherwise waits out the connect timeout on
    every agent build. Cheap to check, and "one of my four servers is not running" is
    routine when developing against a local fleet.
    """
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wrap_mcp_tool_for_langgraph(inner: BaseTool, prefix: str = "") -> BaseTool:
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

        def __init__(self, inner_tool: BaseTool, coro: Any, name: str) -> None:
            super().__init__(
                name=name,
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

    # `__` matches the separator NAT uses to namespace function-group tools, so a tool
    # reads the same whether it arrived through this path or a workflow.yaml block.
    return _MCPToolWrapper(
        inner, _invoke_inner, f"{prefix}__{inner.name}" if prefix else inner.name
    )


@asynccontextmanager
async def mcp_tools_context(
    target: MCPTarget,
    *,
    prefix: str | None = None,
    forwarded: dict[str, str] | None = None,
    auth_context: dict[str, Any] | None = None,
    extra: dict[str, str] | None = None,
    strict: bool = True,
) -> AsyncGenerator[list[BaseTool], None]:
    """Yield the LangChain tools one MCP server exposes.

    Parameters
    ----------
    target : MCPTarget
        The resolved server to connect to, from
        ``build_target(config.resolve_mcp_server(name), ...)``.
    prefix : str | None
        Namespace every tool as ``<prefix>__<tool>``, using the same separator NAT's
        function groups use so traces and evals read the same on both paths. Defaults to
        the server's name. Pass ``""`` to keep the raw names -- safe only with a single
        server, since two servers each exposing ``search`` would otherwise collide and
        one would silently shadow the other.
    forwarded : dict[str, str] | None
        Headers forwarded from the inbound request.
    auth_context : dict[str, Any] | None
        Authorization context to encode for the MCP connection.
    extra : dict[str, str] | None
        Headers merged last, so they override the resolved ones. This is how a caller
        that performs its own token exchange -- Okta cross-application access, say --
        presents the exchanged token, since that flow needs a NAT auth provider and so
        cannot be expressed as the server's `auth_provider` on this path.
    strict : bool
        Raise when the server cannot be reached. The default: a server that was
        configured and is unreachable is a failure, and yielding an empty tool list
        makes it indistinguishable from one that was never configured. Pass ``False``
        for the old degrade-quietly behaviour.
    """
    prefix = target.name if prefix is None else prefix
    server_config = build_server_config(
        target, forwarded=forwarded, auth_context=auth_context, extra=extra
    )

    url = server_config["url"]
    logger.info("Connecting to MCP server %r: %s", target.name, url)

    # A local server that isn't running would otherwise wait out the connect timeout on
    # every start; "one of my four servers is not up" is routine for a local fleet.
    if target.kind is MCPTargetKind.LOCAL and not _local_server_reachable(url):
        message = f"Local MCP server {target.name!r} at {url} is not reachable."
        if strict:
            raise ConnectionError(message)
        logger.warning("%s Continuing without its tools.", message)
        yield []
        return

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
        tools = [_wrap_mcp_tool_for_langgraph(t, prefix) for t in raw_tools]
        logger.info("Loaded %d tools from MCP server %r", len(tools), target.name)
        connected = True
        yield tools
    except (ConnectionError, OSError, TimeoutError, ExceptionGroup) as exc:
        if connected:
            raise
        if strict:
            raise
        logger.warning(
            "Failed to connect to MCP server %r at %s: %s. Continuing without its tools.",
            target.name,
            url,
            exc,
        )
        yield []
