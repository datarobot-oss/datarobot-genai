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

from typing import Any

from llama_index.tools.mcp import BasicMCPClient
from llama_index.tools.mcp import aget_tools_from_mcp_url

from datarobot_genai.core.mcp.common import MCPConfig


async def load_mcp_tools(
    api_base: str | None = None,
    api_key: str | None = None,
) -> list[Any]:
    """
    Asynchronously load MCP tools for LlamaIndex.

    Args:
        api_base: Optional DataRobot API base URL
        api_key: Optional DataRobot API token

    Returns
    -------
        List of MCP tools, or empty list if no MCP configuration is present.
    """
    config = MCPConfig(api_base=api_base, api_key=api_key)
    server_params = config.server_config

    if not server_params:
        print("No MCP server configured, using empty tools list", flush=True)
        return []

    url = server_params["url"]
    headers = server_params.get("headers", {})

    try:
        print(f"Connecting to MCP server: {url}", flush=True)
        # Create BasicMCPClient with headers to pass authentication
        client = BasicMCPClient(command_or_url=url, headers=headers)
        tools = await aget_tools_from_mcp_url(
            command_or_url=url,
            client=client,
        )
        # Ensure list
        tools_list = list(tools) if tools is not None else []
        print(
            f"Successfully connected to MCP server, got {len(tools_list)} tools",
            flush=True,
        )
        return tools_list
    except Exception as e:
        print(
            f"Warning: Failed to connect to MCP server {url}: {e}",
            flush=True,
        )
        return []
