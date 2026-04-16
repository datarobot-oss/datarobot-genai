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

import logging
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from crewai.tools import BaseTool
from crewai_tools import MCPServerAdapter

from datarobot_genai.core.mcp import MCPConfig

logger = logging.getLogger(__name__)


# here it is async to conform with other MCP adapters
@asynccontextmanager
async def mcp_tools_context(mcp_config: MCPConfig) -> AsyncGenerator[list[BaseTool], None]:
    """Context manager for MCP tools that handles connection lifecycle."""
    # If no MCP server configured, return empty tools list
    if not mcp_config.server_config:
        logger.info("No MCP server configured, using empty tools list")
        yield []
        return

    url = mcp_config.server_config["url"]
    logger.info("Connecting to MCP server: %s", url)

    # MCPServerAdapter uses a background thread that prints a huge traceback
    # to stderr when it can't connect. Suppress it so our graceful fallback
    # is the only output the user sees.
    original_excepthook = threading.excepthook

    def _suppress_mcp_thread_exceptions(args: threading.ExceptHookArgs) -> None:
        logger.debug("Suppressed MCPServerAdapter thread exception: %s", args.exc_value)

    try:
        threading.excepthook = _suppress_mcp_thread_exceptions
        with MCPServerAdapter(mcp_config.server_config) as tools:
            threading.excepthook = original_excepthook
            logger.info("Successfully connected to MCP server, got %d tools", len(tools))
            yield tools
    except (ConnectionError, OSError, TimeoutError, ExceptionGroup, RuntimeError) as exc:
        logger.warning(
            "Failed to connect to MCP server at %s: %s. Continuing without MCP tools.",
            url,
            exc,
        )
        yield []
    finally:
        threading.excepthook = original_excepthook
