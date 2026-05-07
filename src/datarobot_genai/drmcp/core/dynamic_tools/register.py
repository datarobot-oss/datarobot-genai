# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Server-bound shim that wires :mod:`datarobot_genai.drtools.dynamic` into drmcp.

The framework-agnostic core (HTTP request callable factory, schema model
generator, deployment adapters) lives in
:mod:`datarobot_genai.drtools.dynamic`. This module remains the public entry
point for drmcp-based servers and preserves the previous import paths.
"""

import logging
from collections.abc import Callable
from collections.abc import Coroutine
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.tools.tool import Tool
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.config import get_config
from datarobot_genai.drmcp.core.mcp_instance import register_tools
from datarobot_genai.drmcp.core.utils import format_response_as_tool_result
from datarobot_genai.drtools.dynamic import external_tool as _drtools_external_tool
from datarobot_genai.drtools.dynamic.external_tool import REQUEST_CONNECT_TIMEOUT
from datarobot_genai.drtools.dynamic.external_tool import REQUEST_FORWARDED_HEADERS
from datarobot_genai.drtools.dynamic.external_tool import REQUEST_MAX_RETRY
from datarobot_genai.drtools.dynamic.external_tool import REQUEST_RETRY_SLEEP
from datarobot_genai.drtools.dynamic.external_tool import REQUEST_RETRYABLE_STATUS_CODES
from datarobot_genai.drtools.dynamic.external_tool import REQUEST_TOTAL_TIMEOUT
from datarobot_genai.drtools.dynamic.external_tool import ExternalToolRegistrationConfig
from datarobot_genai.drtools.dynamic.external_tool import build_external_tool_callable

__all__ = [
    "REQUEST_CONNECT_TIMEOUT",
    "REQUEST_FORWARDED_HEADERS",
    "REQUEST_MAX_RETRY",
    "REQUEST_RETRYABLE_STATUS_CODES",
    "REQUEST_RETRY_SLEEP",
    "REQUEST_TOTAL_TIMEOUT",
    "ExternalToolRegistrationConfig",
    "get_http_headers",
    "get_outbound_headers",
    "register_external_tool",
]

logger = logging.getLogger(__name__)


async def get_outbound_headers(spec: ExternalToolRegistrationConfig) -> dict[str, str]:
    """Drmcp wrapper around :func:`drtools.dynamic.get_outbound_headers`.

    Reads incoming HTTP headers via :func:`fastmcp.server.dependencies.get_http_headers`
    so that drmcp tests can monkeypatch ``register.get_http_headers`` as before.
    Delegates the actual merging logic to the drtools helper to keep behavior in
    sync.
    """
    # Read via the local name so monkeypatching `register.get_http_headers` works.
    headers = get_http_headers()
    # Temporarily swap drtools' header source to the captured incoming headers.
    return await _drtools_external_tool._merge_outbound_headers(spec, headers)


def _external_tool_callable_factory(
    spec: ExternalToolRegistrationConfig,
) -> Callable[[Any], Coroutine[Any, Any, ToolResult]]:
    """Wrap :func:`build_external_tool_callable` with drmcp defaults.

    Kept under its original private name so existing in-tree imports / monkey
    patches continue to work; new code should call
    :func:`datarobot_genai.drtools.dynamic.build_external_tool_callable`
    directly.
    """
    config = get_config()
    return build_external_tool_callable(
        spec,
        format_response=format_response_as_tool_result,
        tool_error_class=ToolError,
        allow_empty_schema=config.tool_registration_allow_empty_schema,
        get_outbound_headers_fn=get_outbound_headers,
    )


async def register_external_tool(config: ExternalToolRegistrationConfig, **kwargs: Any) -> Tool:
    """Create and register a generic HTTP tool in the MCP server.

    This function creates a dynamic tool that makes HTTP requests to external APIs
    and registers it with the MCP server for use by LLM agents.

    Args:
        config: ExternalToolRegistrationConfig object containing all tool parameters.
        **kwargs: Additional keyword arguments to pass to tools registration.

    Returns
    -------
        The registered Tool instance with full MCP integration.

    Raises
    ------
        ValueError: If required path parameters are missing from input_schema.
        aiohttp.ClientError: If the HTTP request fails during tool execution.

    Note:
        The tool remains registered until explicitly removed or the server restarts.
    """
    external_tool_callable = _external_tool_callable_factory(config)

    registered_tool = await register_tools(
        fn=external_tool_callable,
        name=config.name,
        title=config.title,
        description=config.description,
        tags=config.tags,
        **kwargs,
    )

    return registered_tool
