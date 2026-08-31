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

"""
MCP integration for LlamaIndex using llama-index-tools-mcp.

This module provides MCP server connection management for LlamaIndex agents.
Unlike CrewAI which uses a context manager, LlamaIndex uses async calls to
fetch tools from MCP servers.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from llama_index.core.tools import BaseTool
from llama_index.tools.mcp import BasicMCPClient
from llama_index.tools.mcp import aget_tools_from_mcp_url

from datarobot_genai.core.mcp.target import MCPTarget
from datarobot_genai.core.mcp.target import build_server_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def mcp_tools_context(
    target: MCPTarget,
    *,
    prefix: str | None = None,
    forwarded: dict[str, str] | None = None,
    auth_context: dict[str, Any] | None = None,
    strict: bool = True,
) -> AsyncGenerator[list[BaseTool], None]:
    """
    Asynchronously load the LlamaIndex tools one MCP server exposes.

    Args:
        target: The resolved server to connect to.
        prefix: Namespace every tool as ``<prefix>__<tool>``. Defaults to the server's
            name; pass ``""`` to keep raw names, which is safe only with a single server.
        forwarded: Headers forwarded from the inbound request.
        auth_context: Authorization context to encode for the MCP connection.
        strict: Raise when the server cannot be reached, rather than yielding no tools.

    Returns
    -------
        List of MCP tools.
    """
    prefix = target.name if prefix is None else prefix
    server_params = build_server_config(target, forwarded=forwarded, auth_context=auth_context)

    url = server_params["url"]
    headers = server_params.get("headers", {})

    logger.info("Connecting to MCP server %r: %s", target.name, url)

    try:
        # Create BasicMCPClient with headers to pass authentication
        client = BasicMCPClient(command_or_url=url, headers=headers)
        tools = await aget_tools_from_mcp_url(
            command_or_url=url,
            client=client,
        )
        # Ensure list
        tools = list(tools) if tools is not None else []
        if prefix:
            # `__` matches NAT's function-group separator, so two servers exposing the
            # same tool name stay distinct and read the same on both paths.
            for tool in tools:
                tool.metadata.name = f"{prefix}__{tool.metadata.name}"
        logger.info("Loaded %d tools from MCP server %r", len(tools), target.name)
    except (ConnectionError, OSError, TimeoutError, ExceptionGroup) as exc:
        if strict:
            raise
        logger.warning(
            "Failed to connect to MCP server %r at %s: %s. Continuing without its tools.",
            target.name,
            url,
            exc,
        )
        yield []
        return

    yield tools
