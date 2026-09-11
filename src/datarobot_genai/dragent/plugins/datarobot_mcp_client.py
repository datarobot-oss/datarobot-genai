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
from typing import Any
from contextlib import AsyncExitStack

from nat.plugins.mcp.client.client_base import AuthAdapter
from nat.plugins.mcp.client.client_base import MCPStreamableHTTPClient
from nat.plugins.mcp.client.client_config import MCPServerConfig
from nat.plugins.mcp.client.client_impl import MCPClientConfig
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_per_user_function_group
from nat.data_models.component_ref import AuthenticationRef

from pydantic import Field


if TYPE_CHECKING:
    import httpx
    from nat.authentication.interfaces import AuthProviderBase
    from nat.plugins.mcp.client.client_impl import MCPFunctionGroup

logger = logging.getLogger(__name__)


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


def _make_input_schema_enum_safe(tool_fn: Any) -> Any:
    """Workaround for BUZZOK-30556. Configure the registered input schema
    with ``use_enum_values=True`` so NAT's ``_convert_input_pydantic``
    extracts plain strings via ``getattr`` instead of ``Enum`` instances.
    Without this, NAT's downstream ``session_tool.input_schema.model_validate
    (kwargs)`` rejects ``Enum`` instances whose class identity differs from
    the cached enum class (which happens when LangChain builds its own
    ``args_schema`` from the same JSON schema).
    """
    schema = tool_fn.input_schema
    if schema is None or schema is type(None):  # noqa: E721
        return tool_fn
    schema.model_config["use_enum_values"] = True
    schema.model_rebuild(force=True)
    return tool_fn


class DrMcpClientConfig(MCPClientConfig, name="datarobot_mcp_client"):  # type: ignore[call-arg]
    server: MCPServerConfig | None = Field(
        default=None,
        description=(
            "Explicit server to connect to. Omit entirely to auto-discover every "
            "MCP server configured via env vars on agent/config.py's Config."
        ),
    )
    auth_provider: str | AuthenticationRef | None = Field(
        default=None,
        description="Auth provider reference, for the explicit-server case.",
    )


def _already_covered_urls(builder: Builder) -> set[str]:
    """
    URLs already claimed by a sibling `dr_mcp_client` entry that has `server:` set
    explicitly, so the auto-discovery sweep doesn't connect to them a second time.

    Walks whichever function actually owns `tool_names` -- which may be the
    workflow function itself, or (as with streaming_memory_agent) a wrapper one
    level up from it via `inner_agent_name`.
    """
    workflow_config = builder.get_workflow_config()
    inner_name = getattr(workflow_config, "inner_agent_name", None)
    target_config = builder.get_function_config(inner_name) if inner_name else workflow_config
    tool_names = getattr(target_config, "tool_names", [])

    covered: set[str] = set()
    for name in tool_names:
        try:
            fg_config = builder.get_function_group_config(name)
        except Exception:
            continue
        if isinstance(fg_config, DrMcpClientConfig) and fg_config.server is not None:
            covered.add(str(fg_config.server.url).rstrip("/"))
    return covered


@register_per_user_function_group(config_type=DrMcpClientConfig)
async def dr_mcp_client_function_group(config: DrMcpClientConfig, _builder: Builder) -> "MCPFunctionGroup":
    # Local imports: keep NAT plugin discovery (which imports this module just to
    # register the type) from eagerly pulling in the MCP adapter stack or this
    # agent's own Config -- matching the discipline already used elsewhere in this
    # agent's register.py.
    from nat.plugins.mcp.client.client_base import MCPSSEClient  # noqa: PLC0415
    from nat.plugins.mcp.client.client_impl import MCPFunctionGroup  # noqa: PLC0415
    from nat.plugins.mcp.client.client_impl import (  # noqa: PLC0415
        mcp_apply_tool_alias_and_description,
    )
    from nat.plugins.mcp.client.client_impl import mcp_session_tool_function  # noqa: PLC0415

    from datarobot_genai.core.config import resolve_config

    group = MCPFunctionGroup(config=config)

    async def _populate_from_client(client, tool_overrides) -> None:
        all_tools = await client.get_tools()
        overrides = mcp_apply_tool_alias_and_description(all_tools, tool_overrides)
        for tool_name, tool in all_tools.items():
            override = overrides.get(tool_name)
            function_name = override.alias if override and override.alias else tool_name
            description = override.description if override and override.description else tool.description

            tool_fn = _make_input_schema_enum_safe(mcp_session_tool_function(tool, group))
            single_fn = tool_fn.single_fn
            if single_fn is None:
                logger.warning("Skipping tool %s because single_fn is None", function_name)
                continue

            input_schema = tool_fn.input_schema
            if input_schema is type(None):  # noqa: E721
                input_schema = None

            group.add_function(
                name=function_name,
                description=description,
                fn=single_fn,
                input_schema=input_schema,
                converters=tool_fn.converters,
            )

    async def _build_client(url: str, transport: str, auth_provider):
        if transport == "sse":
            return MCPSSEClient(url)
        user_id = getattr(auth_provider.config, "default_user_id", url) if auth_provider else None
        client = DataRobotMCPStreamableHTTPClient(url, auth_provider=auth_provider, user_id=user_id)
        # TODO: for a server with no auth_provider but static headers (the
        # external/third-party case, e.g. weather_mcp_headers), those headers need
        # to reach this client somehow -- not yet wired here. Confirm how the base
        # MCPStreamableHTTPClient / DataRobotMCPStreamableHTTPClient accepts
        # pre-set static headers versus an auth_provider.
        return client

    async with AsyncExitStack() as stack:
        if config.server is not None:
            # Explicit single-server path -- same behavior as the installed
            # plugin's original datarobot_mcp_client_function_group.
            auth_provider = await _builder.get_auth_provider(config.auth_provider) if config.auth_provider else None
            client = await _build_client(str(config.server.url), config.server.transport, auth_provider)
            await stack.enter_async_context(client)
            group.mcp_client = client
            group.mcp_client_server_name = client.server_name
            group.mcp_client_transport = client.transport
            await _populate_from_client(client, config.tool_overrides)
        else:
            # Auto-discovery path -- sweep every MCP server configured via env
            # vars, skipping anything a sibling entry already declared explicitly.
            covered = _already_covered_urls(_builder)
            for mcp_config in resolve_config().resolve_all_mcp_configs():
                url = str(mcp_config.url).rstrip("/")
                if url in covered:
                    logger.debug("Skipping %s: already declared explicitly elsewhere", url)
                    continue

                auth_provider = None
                if getattr(mcp_config, "use_datarobot_auth", False):
                    auth_provider = await _builder.get_auth_provider("datarobot_auth")

                client = await _build_client(
                    str(mcp_config.url),
                    getattr(mcp_config, "transport", "streamable-http"),
                    auth_provider,
                )
                await stack.enter_async_context(client)
                await _populate_from_client(client, tool_overrides={})

        yield group
