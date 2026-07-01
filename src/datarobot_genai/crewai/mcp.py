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
MCP integration for CrewAI.

Loads MCP tools via mcpadapt, preserving each tool's raw ``inputSchema`` so the schema
offered to the LLM stays provider-portable (see ``_RawSchemaCrewAIAdapter``).
"""

import logging
import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

from crewai.tools import BaseTool
from mcpadapt.core import MCPAdapt
from mcpadapt.crewai_adapter import CrewAIAdapter
from pydantic import BaseModel

from datarobot_genai.core.mcp import MCPConfig

logger = logging.getLogger(__name__)

_EMPTY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _local_server_reachable(url: str, timeout: float = 1.0) -> bool:
    """TCP-probe a local MCP server's host:port.

    CrewAI connects via mcpadapt on a background thread, so an unstarted local server
    otherwise blocks ~30s and leaks a thread traceback. A quick probe lets us skip the
    adapter and degrade cleanly instead.
    """
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class _RawSchemaCrewAIAdapter(CrewAIAdapter):
    """Adapt MCP tools but keep the server's raw ``inputSchema`` as the LLM-facing schema.

    The stock adapter rebuilds it through a pydantic model -- a lossy round-trip that drops
    property ``type``s and adds null/empty keys that azure rejects (bedrock tolerates them).
    Keep the model for arg validation; return the raw schema.
    """

    @staticmethod
    def _keep_raw_schema(tool: BaseTool, mcp_tool: Any) -> BaseTool:
        raw = getattr(mcp_tool, "inputSchema", None) or _EMPTY_OBJECT_SCHEMA
        base: type[BaseModel] = tool.args_schema or BaseModel

        class _RawArgsSchema(base):  # type: ignore[valid-type,misc]
            @classmethod
            def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
                return raw

        tool.args_schema = _RawArgsSchema
        return tool

    def adapt(self, func: Any, mcp_tool: Any) -> BaseTool:
        return self._keep_raw_schema(super().adapt(func, mcp_tool), mcp_tool)


@asynccontextmanager
async def mcp_tools_context(mcp_config: MCPConfig) -> AsyncGenerator[list[BaseTool], None]:
    """Context manager for MCP tools that handles connection lifecycle."""
    if not mcp_config.server_config:
        logger.info("No MCP server configured, using empty tools list")
        yield []
        return

    url = mcp_config.server_config["url"]

    # A local MCP server that isn't running would otherwise block ~30s and dump a
    # background-thread traceback; skip the adapter and degrade cleanly.
    if mcp_config.is_local_server and not _local_server_reachable(url):
        logger.warning(
            "Local MCP server at %s is not reachable. Continuing without MCP tools.",
            url,
        )
        yield []
        return

    logger.info("Connecting to MCP server: %s", url)

    try:
        adapter = MCPAdapt(mcp_config.server_config, _RawSchemaCrewAIAdapter())
        tools = adapter.__enter__()
    except Exception as exc:
        logger.warning(
            "Failed to connect to MCP server at %s: %s. Continuing without MCP tools.",
            url,
            exc,
        )
        yield []
        return

    try:
        logger.info("Successfully connected to MCP server, got %d tools", len(tools))
        yield tools
    finally:
        adapter.__exit__(None, None, None)
