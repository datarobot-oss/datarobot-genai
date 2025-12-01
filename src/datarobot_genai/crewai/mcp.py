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
MCP integration for CrewAI using MCPServerAdapter.

This module provides MCP server connection management for CrewAI agents.
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from crewai_tools import MCPServerAdapter

from datarobot_genai.core.mcp.common import MCPConfig


@contextmanager
def mcp_tools_context(
    authorization_context: dict[str, Any] | None = None,
    forwarded_headers: dict[str, str] | None = None,
) -> Generator[list[Any], None, None]:
    """Context manager for MCP tools that handles connection lifecycle."""
    config = MCPConfig(
        authorization_context=authorization_context,
        forwarded_headers=forwarded_headers,
    )
    # If no MCP server configured, return empty tools list
    if not config.server_config:
        print("No MCP server configured, using empty tools list", flush=True)
        yield []
        return

    print(f"Connecting to MCP server: {config.server_config['url']}", flush=True)

    # Use MCPServerAdapter as context manager with the server config
    try:
        with MCPServerAdapter(config.server_config) as tools:
            print(
                f"Successfully connected to MCP server, got {len(tools)} tools",
                flush=True,
            )
            yield tools
    except Exception as exc:
        # Gracefully degrade when connection fails or adapter initialization raises
        print(f"Failed to connect to MCP server: {exc}", flush=True)
        yield []
