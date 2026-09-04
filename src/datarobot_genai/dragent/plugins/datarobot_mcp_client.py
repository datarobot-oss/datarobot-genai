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

import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any

from datarobot.core.config import DEFAULT_MCP_SERVER_NAME
from datarobot.core.config import NO_MCP_AUTH_PROVIDER
from datarobot.core.config import MCPServerRef
from nat.cli.register_workflow import register_per_user_function_group
from nat.data_models.component_ref import AuthenticationRef
from nat.plugins.mcp.client.client_base import AuthAdapter
from nat.plugins.mcp.client.client_base import MCPStreamableHTTPClient
from nat.plugins.mcp.client.client_config import MCPServerConfig
from nat.plugins.mcp.client.client_impl import MCPClientConfig
from nat.plugins.mcp.client.client_impl import MCPFunctionGroup
from nat.plugins.mcp.exception_handler import extract_primary_exception
from pydantic import Field
from pydantic import model_validator

from datarobot_genai.core.mcp.target import MCPTarget
from datarobot_genai.core.mcp.target import build_target

if TYPE_CHECKING:
    import httpx
    from nat.authentication.interfaces import AuthProviderBase
    from nat.builder.builder import Builder

logger = logging.getLogger(__name__)

#: Address fields NAT's base server config declares that this client does not read.
#: An inherited field that is silently ignored is worse than one that raises.
_INHERITED_ADDRESS_FIELDS = ("url", "command", "args", "env", "custom_headers")


class DataRobotMCPServerConfig(MCPServerConfig):
    """Which configured MCP server a block connects to, and as whom.

    Names a server and an identity. The *address* never appears here: it is declared in
    ``MCP_SERVERS`` and resolved off the application's config, so the same
    ``workflow.yaml`` works on a laptop and in production, and an address created by a
    deploy resolves once the deploy has run.
    """

    name: str = Field(
        default=DEFAULT_MCP_SERVER_NAME,
        description=(
            "Which configured MCP server this block connects to, by name. The address "
            "lives in MCP_SERVERS, never here."
        ),
    )
    # Authentication configuration
    auth_provider: str | AuthenticationRef | None = Field(
        default="datarobot_mcp_auth",
        description="Reference to authentication provider",
    )

    @model_validator(mode="after")
    def validate_model(self) -> DataRobotMCPServerConfig:
        """Reject an inline address, replacing NAT's validator that demands one.

        Deliberately shadows ``MCPServerConfig.validate_model`` by name, which is how
        pydantic replaces an inherited validator. NAT's version requires ``url`` for
        every non-stdio transport -- correct for a client whose config carries the
        address, and exactly wrong for one whose address comes from ``MCP_SERVERS``.
        """
        offenders = [f for f in _INHERITED_ADDRESS_FIELDS if getattr(self, f, None) is not None]
        # `transport` needs model_fields_set because its default is non-None, so unset
        # and explicitly-set-to-the-default are otherwise indistinguishable.
        if "transport" in self.model_fields_set:
            offenders.append("transport")
        if offenders:
            raise ValueError(
                f"MCP block {self.name!r} sets {offenders}, which this client ignores. "
                f"Put the address in MCP_SERVERS and keep only `name` and `auth_provider` "
                f"here."
            )
        return self


class DataRobotMCPClientConfig(MCPClientConfig, name="datarobot_mcp_client"):  # type: ignore[call-arg]
    server: DataRobotMCPServerConfig = Field(
        default_factory=DataRobotMCPServerConfig,
        description="DataRobot MCP Server configuration",
    )


def resolve_auth_provider_name(
    server: DataRobotMCPServerConfig, ref: MCPServerRef
) -> str | AuthenticationRef | None:
    """Decide which auth provider this block uses, from the two places it may be set.

    ``workflow.yaml``'s ``server.auth_provider`` and the ref's ``auth_provider`` (from
    ``<name>_mcp_auth_provider``) are two sources for one value. Setting both is an
    error rather than a precedence rule -- there is no correct answer to pick, and
    picking one silently reintroduces exactly the kind of hidden layer this design
    removes.

    ``model_fields_set`` is what distinguishes an explicit YAML value from the field's
    default, which is non-``None`` and so otherwise always looks set.
    """
    yaml_set = "auth_provider" in server.model_fields_set
    ref_set = ref.auth_provider is not None

    if yaml_set and ref_set and str(server.auth_provider) != ref.auth_provider:
        raise ValueError(
            f"MCP server {ref.name!r} has an auth provider in two places: "
            f"workflow.yaml says {server.auth_provider!r} and "
            f"{ref.name}_mcp_auth_provider says {ref.auth_provider!r}. Set it in one "
            f"place; there is no precedence between them."
        )

    name = ref.auth_provider if ref_set and not yaml_set else server.auth_provider
    if name is None or name == NO_MCP_AUTH_PROVIDER:
        return None
    return name


class DataRobotAuthAdapter(AuthAdapter):
    """Carries this connection's resolved target through to the shared auth provider.

    NAT hands one auth provider instance to every block that names it, so the provider
    cannot hold the target. The adapter is per-client, so it can.
    """

    def __init__(
        self,
        auth_provider: AuthProviderBase,
        user_id: str | None,
        target: MCPTarget | None = None,
    ) -> None:
        super().__init__(auth_provider, user_id)
        self._target = target

    async def _get_auth_headers(
        self, request: httpx.Request | None = None, response: httpx.Response | None = None
    ) -> dict[str, str]:
        """Get authentication headers from the NAT auth provider."""
        try:
            # Use the user_id passed to this AuthAdapter instance
            auth_result = await self.auth_provider.authenticate(
                user_id=self.user_id, response=response, target=self._target
            )
            as_kwargs = auth_result.as_requests_kwargs()
            return as_kwargs["headers"]
        except (ValueError, TypeError):
            # A missing or malformed target is a wiring bug, not a credential failure.
            # Swallowing it here would hand back an unauthenticated client, which is
            # precisely the silent degradation this design removes.
            raise
        except Exception as e:
            logger.warning("Failed to get auth token: %s", e)
            return {}


class DataRobotMCPStreamableHTTPClient(MCPStreamableHTTPClient):
    def __init__(
        self,
        url: str,
        auth_provider: AuthProviderBase | None = None,
        user_id: str | None = None,
        target: MCPTarget | None = None,
        custom_headers: dict[str, str] | None = None,
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
            custom_headers=custom_headers,
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
            DataRobotAuthAdapter(auth_provider, effective_user_id, target)
            if auth_provider
            else None
        )


class DataRobotMCPFunctionGroup(MCPFunctionGroup):  # type: ignore[misc]
    """An MCP function group that remembers which server it resolved to.

    ``_target`` is the block's own resolved server. It is set at build time and read by
    every client this group creates, so credentials follow the block's kind rather than
    whatever the environment happened to resolve.

    Per-user session clients are deliberately unsupported. NAT builds those in
    ``_create_session_client`` from ``config.server.url`` -- a field this client no longer
    has, because the address comes from ``MCP_SERVERS`` -- and wires them to NAT's own
    ``AuthAdapter``, which carries no target and would make our auth provider raise. An
    earlier version reimplemented that factory, which meant duplicating NAT's session
    lifetime logic (the ready/stop events and the ``_lifetime`` task, whose cancel scope
    must be entered and exited in the same task). That is replaced by the guard below.

    The guard is currently unreachable. ``_get_session_id_from_context`` only returns a
    session id when the request carries a ``nat-session`` cookie, NAT issues that cookie
    only from its WebSocket route, and ``DRAgentFastApiFrontEndConfig`` registers no
    ``websocket_path`` (see ``dragent/frontends/claim_validation.py``). So every call
    falls through to ``_default_user_id`` and uses the build-time client. The guard exists
    because that is a property of the front-end configuration, not of MCP: if a WebSocket
    route is ever registered, this path becomes live, and a loud failure naming the fix is
    much better than the alternative -- NAT's base implementation would build a client
    pointed at the string ``"None"`` with no credentials, and every tool call would return
    "Tool temporarily unavailable" to the model while the agent reported healthy.
    """

    _target: MCPTarget | None = None

    async def _create_session_client(
        self, session_id: str
    ) -> tuple[Any, asyncio.Event, asyncio.Task[None]]:
        """Refuse to create a per-user session client. See the class docstring."""
        raise RuntimeError(
            "datarobot_mcp_client does not support per-user MCP session clients. The "
            "server address comes from MCP_SERVERS rather than from workflow.yaml, so "
            "NAT's session factory has no URL to connect to and its auth adapter cannot "
            "carry this block's target. Set `session_aware_tools: false` on this function "
            "group so tool calls use the build-time client, which resolves both."
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


@register_per_user_function_group(config_type=DataRobotMCPClientConfig)
async def datarobot_mcp_client_function_group(
    config: DataRobotMCPClientConfig, _builder: Builder
) -> AsyncGenerator[DataRobotMCPFunctionGroup]:
    """
    Connect to an MCP server and expose tools as a function group.

    Args:
        config: The configuration for the MCP client
        _builder: The builder
    Returns:
        The function group
    """
    from nat.plugins.mcp.client.client_base import MCPSSEClient  # noqa: PLC0415
    from nat.plugins.mcp.client.client_impl import (
        mcp_apply_tool_alias_and_description,  # noqa: PLC0415
    )
    from nat.plugins.mcp.client.client_impl import mcp_session_tool_function  # noqa: PLC0415

    from datarobot_genai.core.config import resolve_config  # noqa: PLC0415

    # Resolved once, at build, for this block alone. Raising here fails the build naming
    # the server, instead of yielding a client pointed at a dead port carrying no
    # credentials -- which is what an unresolvable address used to produce.
    app_config = resolve_config()
    target = build_target(
        app_config.resolve_mcp_server(config.server.name),
        datarobot_endpoint=app_config.resolve_datarobot_endpoint(),
        datarobot_api_token=app_config.resolve_datarobot_api_token(),
    )

    # Which identity, reconciled between the two places it may be declared. Setting it
    # in both is an error, not a precedence rule: there is no correct answer to pick,
    # and choosing one silently is the failure this design removes.
    provider_name = resolve_auth_provider_name(config.server, target.ref)

    # Resolve auth provider if specified. NAT returns ONE SHARED instance per name, so
    # every block naming `datarobot_mcp_auth` gets the same object -- which is why the
    # target travels with each call rather than being stored on the provider.
    auth_provider = None
    if provider_name:
        auth_provider = await _builder.get_auth_provider(provider_name)

    # Build the appropriate client
    if target.ref.transport == "sse":
        client = MCPSSEClient(
            target.url,
            tool_call_timeout=config.tool_call_timeout,
            auth_flow_timeout=config.auth_flow_timeout,
            reconnect_enabled=config.reconnect_enabled,
            reconnect_max_attempts=config.reconnect_max_attempts,
            reconnect_initial_backoff=config.reconnect_initial_backoff,
            reconnect_max_backoff=config.reconnect_max_backoff,
        )
    else:
        # Use default_user_id for the base client
        # For interactive OAuth2: from config. For service accounts: defaults to server URL
        base_user_id = (
            getattr(auth_provider.config, "default_user_id", target.url) if auth_provider else None
        )
        client = DataRobotMCPStreamableHTTPClient(
            target.url,
            auth_provider=auth_provider,
            user_id=base_user_id,
            target=target,
            tool_call_timeout=config.tool_call_timeout,
            auth_flow_timeout=config.auth_flow_timeout,
            reconnect_enabled=config.reconnect_enabled,
            reconnect_max_attempts=config.reconnect_max_attempts,
            reconnect_initial_backoff=config.reconnect_initial_backoff,
            reconnect_max_backoff=config.reconnect_max_backoff,
        )

    logger.info("MCP server %r (%s) is at %s", target.name, target.kind.value, client.server_name)

    # Create the MCP function group
    group = DataRobotMCPFunctionGroup(config=config)

    # Store shared components for session client creation
    group._shared_auth_provider = auth_provider
    group._client_config = config
    # Per-block, and read by the per-user session factory: the per-user client does not
    # inherit the build-time client's target.
    group._target = target

    # Set auth provider config defaults
    # For interactive OAuth2: use config values
    # For service accounts: default_user_id = server URL,
    #                       allow_default_user_id_for_tool_calls = True
    if auth_provider:
        group._default_user_id = getattr(auth_provider.config, "default_user_id", target.url)
        group._allow_default_user_id_for_tool_calls = getattr(
            auth_provider.config, "allow_default_user_id_for_tool_calls", True
        )
    else:
        group._default_user_id = None
        group._allow_default_user_id_for_tool_calls = True

    yielded = False
    try:
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

                # Create the tool function according to configuration.
                # Patch the input schema to store enum values as strings so
                # NAT's downstream model_validate(kwargs) never trips on
                # cross-class enum identity (BUZZOK-30556).
                tool_fn = _make_input_schema_enum_safe(mcp_session_tool_function(tool, group))

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
                logger.debug("Adding tool %s to group", function_name)
                group.add_function(
                    name=function_name,
                    description=description,
                    fn=single_fn,
                    input_schema=input_schema,
                    converters=tool_fn.converters,
                )
            yielded = True
            yield group
    except Exception as e:
        if hasattr(e, "exceptions"):
            primary_exception = extract_primary_exception(list(e.exceptions))
        else:
            primary_exception = e

        logger.warning("Error in MCP client function group: %s", primary_exception)
        group.mcp_client = None
        group.mcp_client_server_name = getattr(client, "server_name", target.url)
        group.mcp_client_transport = getattr(client, "transport", target.ref.transport)
        if not yielded:
            yield group
        else:
            # Cleanup (e.g. __aexit__) failed after we already yielded; re-raise so
            # the caller sees it and we do not yield a second time.
            raise
