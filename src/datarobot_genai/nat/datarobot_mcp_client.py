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

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Literal

from nat.cli.register_workflow import register_function_group
from nat.data_models.component_ref import AuthenticationRef
from nat.plugins.mcp.client_base import AuthAdapter
from nat.plugins.mcp.client_base import MCPStreamableHTTPClient
from nat.plugins.mcp.client_config import MCPServerConfig
from nat.plugins.mcp.client_impl import MCPClientConfig
from pydantic import Field
from pydantic import HttpUrl

if TYPE_CHECKING:
    import httpx
    from nat.authentication.interfaces import AuthProviderBase
    from nat.builder.builder import Builder
    from nat.plugins.mcp.client_impl import MCPFunctionGroup

logger = logging.getLogger(__name__)


def _default_transport() -> Literal["streamable-http", "sse", "stdio"]:
    from datarobot_genai.core.mcp.common import MCPConfig  # noqa: PLC0415

    server_config = MCPConfig().server_config
    return server_config["transport"] if server_config else "stdio"


def _default_url() -> HttpUrl | None:
    from datarobot_genai.core.mcp.common import MCPConfig  # noqa: PLC0415

    server_config = MCPConfig().server_config
    return server_config["url"] if server_config else None


def _default_auth_provider() -> str | AuthenticationRef | None:
    from datarobot_genai.core.mcp.common import MCPConfig  # noqa: PLC0415

    server_config = MCPConfig().server_config
    return "datarobot_mcp_auth" if server_config else None


def _default_command() -> str | None:
    from datarobot_genai.core.mcp.common import MCPConfig  # noqa: PLC0415

    server_config = MCPConfig().server_config
    return None if server_config else "docker"


class DataRobotMCPServerConfig(MCPServerConfig):
    transport: Literal["streamable-http", "sse", "stdio"] = Field(
        default_factory=_default_transport,
        description="Transport type to connect to the MCP server (sse or streamable-http)",
    )
    url: HttpUrl | None = Field(
        default_factory=_default_url,
        description="URL of the MCP server (for sse or streamable-http transport)",
    )
    # Authentication configuration
    auth_provider: str | AuthenticationRef | None = Field(
        default_factory=_default_auth_provider,
        description="Reference to authentication provider",
    )
    command: str | None = Field(
        default_factory=_default_command,
        description="Command to run for stdio transport (e.g. 'python' or 'docker')",
    )


class DataRobotMCPClientConfig(MCPClientConfig, name="datarobot_mcp_client"):  # type: ignore[call-arg]
    server: DataRobotMCPServerConfig = Field(
        default_factory=DataRobotMCPServerConfig,
        description="DataRobot MCP Server configuration",
    )


class DataRobotAuthAdapter(AuthAdapter):
    async def _get_auth_headers(
        self, request: httpx.Request | None = None, response: httpx.Response | None = None
    ) -> dict[str, str]:
        """Get authentication headers from the NAT auth provider."""
        try:
            # Use the user_id passed to this AuthAdapter instance
            auth_result = await self.auth_provider.authenticate(
                user_id=self.user_id, response=response
            )
            as_kwargs = auth_result.as_requests_kwargs()
            return as_kwargs["headers"]
        except Exception as e:
            logger.warning("Failed to get auth token: %s", e)
            return {}


class DataRobotMCPStreamableHTTPClient(MCPStreamableHTTPClient):
    def __init__(
        self,
        url: str,
        auth_provider: AuthProviderBase | None = None,
        user_id: str | None = None,
        tool_call_timeout: timedelta = timedelta(seconds=60),
        auth_flow_timeout: timedelta = timedelta(seconds=300),
        reconnect_enabled: bool = True,
        reconnect_max_attempts: int = 2,
        reconnect_initial_backoff: float = 0.5,
        reconnect_max_backoff: float = 50.0,
    ):
        super().__init__(
            url=url,
            auth_provider=auth_provider,
            user_id=user_id,
            tool_call_timeout=tool_call_timeout,
            auth_flow_timeout=auth_flow_timeout,
            reconnect_enabled=reconnect_enabled,
            reconnect_max_attempts=reconnect_max_attempts,
            reconnect_initial_backoff=reconnect_initial_backoff,
            reconnect_max_backoff=reconnect_max_backoff,
        )
        effective_user_id = user_id or (
            auth_provider.config.default_user_id if auth_provider else None
        )
        self._httpx_auth = (
            DataRobotAuthAdapter(auth_provider, effective_user_id) if auth_provider else None
        )


@register_function_group(config_type=DataRobotMCPClientConfig)
async def datarobot_mcp_client_function_group(
    config: DataRobotMCPClientConfig, _builder: Builder
) -> MCPFunctionGroup:
    """
    Connect to an MCP server and expose tools as a function group.

    Args:
        config: The configuration for the MCP client
        _builder: The builder
    Returns:
        The function group
    """
    from nat.plugins.mcp.client_base import MCPSSEClient  # noqa: PLC0415
    from nat.plugins.mcp.client_base import MCPStdioClient  # noqa: PLC0415
    from nat.plugins.mcp.client_impl import MCPFunctionGroup  # noqa: PLC0415
    from nat.plugins.mcp.client_impl import mcp_apply_tool_alias_and_description  # noqa: PLC0415
    from nat.plugins.mcp.client_impl import mcp_session_tool_function  # noqa: PLC0415

    # Resolve auth provider if specified
    auth_provider = None
    if config.server.auth_provider:
        auth_provider = await _builder.get_auth_provider(config.server.auth_provider)

    # Build the appropriate client
    if config.server.transport == "stdio":
        if not config.server.command:
            raise ValueError("command is required for stdio transport")
        client = MCPStdioClient(
            config.server.command,
            config.server.args,
            config.server.env,
            tool_call_timeout=config.tool_call_timeout,
            auth_flow_timeout=config.auth_flow_timeout,
            reconnect_enabled=config.reconnect_enabled,
            reconnect_max_attempts=config.reconnect_max_attempts,
            reconnect_initial_backoff=config.reconnect_initial_backoff,
            reconnect_max_backoff=config.reconnect_max_backoff,
        )
    elif config.server.transport == "sse":
        client = MCPSSEClient(
            str(config.server.url),
            tool_call_timeout=config.tool_call_timeout,
            auth_flow_timeout=config.auth_flow_timeout,
            reconnect_enabled=config.reconnect_enabled,
            reconnect_max_attempts=config.reconnect_max_attempts,
            reconnect_initial_backoff=config.reconnect_initial_backoff,
            reconnect_max_backoff=config.reconnect_max_backoff,
        )
    elif config.server.transport == "streamable-http":
        # Use default_user_id for the base client
        base_user_id = auth_provider.config.default_user_id if auth_provider else None
        client = DataRobotMCPStreamableHTTPClient(
            str(config.server.url),
            auth_provider=auth_provider,
            user_id=base_user_id,
            tool_call_timeout=config.tool_call_timeout,
            auth_flow_timeout=config.auth_flow_timeout,
            reconnect_enabled=config.reconnect_enabled,
            reconnect_max_attempts=config.reconnect_max_attempts,
            reconnect_initial_backoff=config.reconnect_initial_backoff,
            reconnect_max_backoff=config.reconnect_max_backoff,
        )
    else:
        raise ValueError(f"Unsupported transport: {config.server.transport}")

    logger.info("Configured to use MCP server at %s", client.server_name)

    # Create the MCP function group
    group = MCPFunctionGroup(config=config)

    # Store shared components for session client creation
    group._shared_auth_provider = auth_provider
    group._client_config = config

    async with client:
        # Expose the live MCP client on the function group instance so other components
        # (e.g., HTTP endpoints) can reuse the already-established session instead of creating a
        # new client per request.
        group.mcp_client = client
        group.mcp_client_server_name = client.server_name
        group.mcp_client_transport = client.transport

        all_tools = await client.get_tools()
        tool_overrides = mcp_apply_tool_alias_and_description(all_tools, config.tool_overrides)

        # Add each tool as a function to the group
        for tool_name, tool in all_tools.items():
            # Get override if it exists
            override = tool_overrides.get(tool_name)

            # Use override values or defaults
            function_name = override.alias if override and override.alias else tool_name
            description = (
                override.description if override and override.description else tool.description
            )

            # Create the tool function according to configuration
            tool_fn = mcp_session_tool_function(tool, group)

            # Normalize optional typing for linter/type-checker compatibility
            single_fn = tool_fn.single_fn
            if single_fn is None:
                # Should not happen because FunctionInfo always sets a single_fn
                logger.warning("Skipping tool %s because single_fn is None", function_name)
                continue

            input_schema = tool_fn.input_schema
            # Convert NoneType sentinel to None for FunctionGroup.add_function signature
            if input_schema is type(None):  # noqa: E721
                input_schema = None

            # Add to group
            logger.info("Adding tool %s to group", function_name)
            group.add_function(
                name=function_name,
                description=description,
                fn=single_fn,
                input_schema=input_schema,
                converters=tool_fn.converters,
            )

        yield group
