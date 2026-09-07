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

Loads MCP tools via mcpadapt, keeping each tool's server ``inputSchema`` as the LLM-facing
function-call parameters so the schema stays provider-portable (see ``_RawSchemaCrewAIAdapter``).
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

from datarobot_genai.core.mcp.target import MCPTarget
from datarobot_genai.core.mcp.target import MCPTargetKind
from datarobot_genai.core.mcp.target import build_server_config

logger = logging.getLogger(__name__)

_EMPTY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _local_server_reachable(url: str, timeout: float = 1.0) -> bool:
    """TCP-probe a local MCP server's host:port.

    CrewAI connects via mcpadapt on a background thread, so an
    unstarted local server otherwise blocks ~30s and leaks a thread traceback.
    A quick probe lets us skip the adapter and degrade cleanly instead.
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
    """Adapt MCP tools but hand the LLM the server's ``inputSchema`` as the tool parameters.

    The stock adapter derives the function-call ``parameters`` from a pydantic model -- a
    lossy round-trip that drops property ``type``s and adds null/empty keys that azure rejects
    (bedrock tolerates them). Keep the model for arg validation, but return the server's schema
    (as ``super().adapt`` leaves it -- ``$ref``s resolved) for the ``parameters``. Note the
    text-prompt ``description`` still carries the lossy schema; the native tool-call path uses
    ``parameters``.
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


# here it is async to conform with other MCP adapters
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
    """Yield the CrewAI tools one MCP server exposes, managing the connection lifecycle.

    Parameters
    ----------
    target : MCPTarget
        The resolved server to connect to.
    prefix : str | None
        Namespace every tool as ``<prefix>__<tool>``. Defaults to the server's name;
        pass ``""`` to keep raw names, which is safe only with a single server.
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
        Raise when the server cannot be reached, rather than yielding no tools.
    """
    prefix = target.name if prefix is None else prefix
    server_config = build_server_config(
        target, forwarded=forwarded, auth_context=auth_context, extra=extra
    )
    url = server_config["url"]

    # A local MCP server that isn't running would otherwise block ~30s and dump
    # a background-thread traceback; skip the adapter rather than wait it out.
    if target.kind is MCPTargetKind.LOCAL and not _local_server_reachable(url):
        message = f"Local MCP server {target.name!r} at {url} is not reachable."
        if strict:
            raise ConnectionError(message)
        logger.warning("%s Continuing without its tools.", message)
        yield []
        return

    logger.info("Connecting to MCP server %r: %s", target.name, url)

    try:
        adapter = MCPAdapt(server_config, _RawSchemaCrewAIAdapter())
        tools = adapter.__enter__()
    except Exception as exc:
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

    try:
        if prefix:
            # `__` matches NAT's function-group separator, so two servers exposing the
            # same tool name stay distinct and read the same on both paths.
            for tool in tools:
                tool.name = f"{prefix}__{tool.name}"
        logger.info("Loaded %d tools from MCP server %r", len(tools), target.name)
        yield tools
    finally:
        adapter.__exit__(None, None, None)
