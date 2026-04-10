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

from llama_index.core.tools import BaseTool
from llama_index.tools.mcp import BasicMCPClient
from llama_index.tools.mcp import aget_tools_from_mcp_url

from datarobot_genai.core.mcp import MCPConfig

logger = logging.getLogger(__name__)


@asynccontextmanager
async def mcp_tools_context(
    mcp_config: MCPConfig,
) -> AsyncGenerator[list[BaseTool], None]:
    """
    Asynchronously load MCP tools for LlamaIndex.

    Args:
        authorization_context: Optional authorization context for MCP connections
        forwarded_headers: Optional forwarded headers, e.g. x-datarobot-api-key for MCP auth

    Returns
    -------
        List of MCP tools, or empty list if no MCP configuration is present.
    """
    server_params = mcp_config.server_config

    if not server_params:
        logger.info("No MCP server configured, using empty tools list")
        yield []
        return

    url = server_params["url"]
    headers = server_params.get("headers", {})

    logger.info(f"Connecting to MCP server: {url}")
    # Create BasicMCPClient with headers to pass authentication
    client = BasicMCPClient(command_or_url=url, headers=headers)
    tools = await aget_tools_from_mcp_url(
        command_or_url=url,
        client=client,
    )
    # Ensure list
    tools = list(tools) if tools is not None else []
    logger.info(f"Successfully connected to MCP server, got {len(tools)} tools")

    yield tools
