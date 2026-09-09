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
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from datarobot.core.config import DEFAULT_MCP_SERVER_NAME
from datarobot.core.config import NO_MCP_AUTH_PROVIDER
from datarobot.core.config import MCPServerRef
from nat.cli.register_workflow import register_per_user_function_group
from nat.data_models.component_ref import AuthenticationRef
from nat.plugins.mcp.client.client_base import MCPSSEClient
from nat.plugins.mcp.client.client_base import MCPStdioClient
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
    from nat.builder.builder import Builder

logger = logging.getLogger(__name__)

#: Fields that configure a *stdio* server, as NAT documents them.
_STDIO_ONLY_FIELDS = ("command", "args", "env")


class DataRobotMCPServerConfig(MCPServerConfig):
    """Which MCP server a block connects to, and as whom.

    Three ways to say where the server is, and **exactly one of them per server**:

    * ``name`` -- the usual one. The address is declared in the environment as
      ``<name>_mcp_<field>`` and resolved off the application's config, so the same
      ``workflow.yaml`` works on a laptop and in production, and an address created by
      a deploy resolves once the deploy has run.
    * ``url`` -- inline, as NAT documents it. Right for an address that is genuinely
      identical in every environment (a pinned third-party endpoint), and for a demo or
      integration test you want self-contained.
    * ``command`` / ``args`` / ``env`` with ``transport: stdio`` -- a local MCP process
      this agent launches, exactly as NAT documents it. A stdio server has no URL and no
      DataRobot identity: it is a child process, so ``auth_provider`` does not apply
      (NAT supports it for ``streamable-http`` only) and none of the fleet resolution
      runs.

    Naming a server that the environment also configures is an error rather than a
    precedence rule: there is no correct answer to pick, and picking one silently is the
    class of bug this design exists to remove. That check needs the resolved fleet, so
    it lives in the builder rather than here.
    """

    name: str = Field(
        default=DEFAULT_MCP_SERVER_NAME,
        description=(
            "Which configured MCP server this block connects to, by name. Its address "
            "comes from <name>_mcp_* in the environment. Leave `url` unset to use this."
        ),
    )
    # Authentication configuration
    auth_provider: str | AuthenticationRef | None = Field(
        default="datarobot_mcp_auth",
        description="Reference to authentication provider",
    )

    @property
    def is_stdio(self) -> bool:
        """Whether this block launches a local MCP process rather than dialling one."""
        return self.transport == "stdio" or self.command is not None

    @model_validator(mode="after")
    def validate_model(self) -> DataRobotMCPServerConfig:
        """Check the address forms are coherent, without requiring any one of them.

        Deliberately shadows ``MCPServerConfig.validate_model`` by name, which is how
        pydantic replaces an inherited validator. NAT's version *requires* ``url`` for
        every non-stdio transport; here it is optional, because the address may instead
        come from the environment under ``name``.
        """
        if self.transport == "stdio" and self.command is None:
            raise ValueError(
                f"MCP block {self.name!r} sets `transport: stdio` but no `command` to run."
            )
        if self.command is None:
            # `args`/`env` only mean anything alongside a command. Ignoring them silently
            # is how a block ends up looking configured and behaving otherwise.
            orphaned = [f for f in ("args", "env") if getattr(self, f, None) is not None]
            if orphaned:
                raise ValueError(
                    f"MCP block {self.name!r} sets {orphaned} without a `command`. Those "
                    f"configure a stdio server; add `command` and `transport: stdio`, or "
                    f"remove them."
                )
        if self.command is not None and self.transport != "stdio":
            raise ValueError(
                f"MCP block {self.name!r} sets `command`, which launches a local MCP "
                f"process, but `transport: {self.transport}`. Set `transport: stdio`."
            )
        if self.is_stdio and self.url is not None:
            raise ValueError(
                f"MCP block {self.name!r} sets both `command` and `url`. A stdio server "
                f"is a child process and has no URL."
            )
        # NAT supports `auth_provider` for streamable-http only, and a child process has
        # no identity to present anyway. The default is non-None, so only an explicit
        # value counts.
        if self.is_stdio and "auth_provider" in self.model_fields_set and self.auth_provider:
            raise ValueError(
                f"MCP block {self.name!r} is a stdio server but sets "
                f"`auth_provider: {self.auth_provider}`. Authentication providers apply to "
                f"`streamable-http` servers only."
            )
        return self


class DataRobotMCPClientConfig(MCPClientConfig, name="datarobot_mcp_client"):  # type: ignore[call-arg]
    server: DataRobotMCPServerConfig = Field(
        default_factory=DataRobotMCPServerConfig,
        description="DataRobot MCP Server configuration",
    )


def resolve_server_ref(server: DataRobotMCPServerConfig, app_config: Any) -> MCPServerRef:
    """Resolve which server this block addresses, from the one source that declares it.

    An inline ``url`` makes the block self-contained: it is not looked up in the fleet,
    and `transport`/`custom_headers` on the block travel with it. Otherwise the block's
    ``name`` is resolved against the environment-configured fleet.

    Declaring a server in *both* places is an error rather than a precedence rule. It is
    only reachable when the block explicitly names a server, because the default name
    (``default``) is not a name anyone chose -- the application templates set
    ``MCP_SERVER_PORT`` unconditionally, so treating that synthesised ``default`` as a
    competing definition would break a stock template the moment it added an inline URL.
    """
    if server.url is None:
        return cast(MCPServerRef, app_config.resolve_mcp_server(server.name))

    named_explicitly = "name" in server.model_fields_set
    if named_explicitly:
        try:
            app_config.resolve_mcp_server(server.name)
        except LookupError:
            pass
        else:
            raise ValueError(
                f"MCP server {server.name!r} is defined twice: `url` is set on the "
                f"workflow.yaml block and {server.name}_mcp_* is set in the environment. "
                f"There is no precedence between them -- remove one."
            )

    return MCPServerRef(
        name=server.name,
        url=str(server.url).rstrip("/"),
        transport=server.transport,
        headers=dict(server.custom_headers or {}),
    )


def resolve_auth_provider_name(
    server: DataRobotMCPServerConfig, ref: MCPServerRef
) -> str | AuthenticationRef | None:
    """Decide which auth provider this block uses, from the two places it may be set.

    ``workflow.yaml``'s ``server.auth_provider`` and ``<name>_mcp_auth_provider`` are two
    sources for one value; setting both is an error rather than a precedence rule.
    ``model_fields_set`` distinguishes an explicit YAML value from the non-``None``
    default.
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


class DataRobotMCPFunctionGroup(MCPFunctionGroup):  # type: ignore[misc]
    """Kept only as a stable name for this client's function group.

    It adds nothing. Everything that used to live here -- a per-block ``_target``, an
    auth adapter that carried it, a client subclass to install that adapter, and an
    override of NAT's per-user session factory -- existed to tell one shared auth
    provider which server was asking. Nothing varies per server any more (see
    ``build_datarobot_mcp_headers``), so the provider is self-sufficient, NAT's own
    client and session factory work unchanged, and all of that is gone.
    """


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
    from nat.plugins.mcp.client.client_impl import (
        mcp_apply_tool_alias_and_description,  # noqa: PLC0415
    )
    from nat.plugins.mcp.client.client_impl import mcp_session_tool_function  # noqa: PLC0415

    from datarobot_genai.core.config import resolve_config  # noqa: PLC0415

    timeouts = {
        "tool_call_timeout": config.tool_call_timeout,
        "auth_flow_timeout": config.auth_flow_timeout,
        "reconnect_enabled": config.reconnect_enabled,
        "reconnect_max_attempts": config.reconnect_max_attempts,
        "reconnect_initial_backoff": config.reconnect_initial_backoff,
        "reconnect_max_backoff": config.reconnect_max_backoff,
    }

    target: MCPTarget | None = None
    auth_provider = None

    if config.server.is_stdio:
        # A local child process: no URL to resolve, no fleet to consult, and no identity
        # to present. None of the credential machinery below applies to it, which is why
        # it is a separate branch rather than a null target threaded through that code.
        assert config.server.command is not None  # guaranteed by validate_model
        client = MCPStdioClient(
            command=config.server.command,
            args=config.server.args,
            env=config.server.env,
            **timeouts,
        )
        logger.info(
            "MCP server %r is a local stdio process: %s", config.server.name, config.server.command
        )
    else:
        # Resolved once, at build. Raising here fails the build naming the server,
        # rather than yielding a client pointed at a dead port.
        app_config = resolve_config()
        target = build_target(
            resolve_server_ref(config.server, app_config),
            datarobot_endpoint=app_config.resolve_datarobot_endpoint(),
            datarobot_api_token=app_config.resolve_datarobot_api_token(),
        )

        # Which identity, reconciled between the two places it may be declared. Setting
        # it in both is an error, not a precedence rule: there is no correct answer to
        # pick, and choosing one silently is the failure this design removes.
        provider_name = resolve_auth_provider_name(config.server, target.ref)

        # NAT returns ONE SHARED instance per name, so every block naming
        # `datarobot_mcp_auth` gets the same object -- which is why the target travels
        # with each call rather than being stored on the provider.
        if provider_name:
            auth_provider = await _builder.get_auth_provider(provider_name)

        # Write the resolved address back onto the block. NAT's own code -- notably its
        # per-user session factory -- reads `config.server.url`, so an address that came
        # from the environment has to land here for those paths to work unchanged.
        config.server.url = target.url  # type: ignore[assignment]

        if target.ref.transport == "sse":
            client = MCPSSEClient(target.url, **timeouts)
        else:
            # Use default_user_id for the base client
            # For interactive OAuth2: from config. For service accounts: the server URL
            base_user_id = (
                getattr(auth_provider.config, "default_user_id", target.url)
                if auth_provider
                else None
            )
            client = MCPStreamableHTTPClient(
                target.url,
                auth_provider=auth_provider,
                user_id=base_user_id,
                custom_headers=config.server.custom_headers,
                **timeouts,
            )

        logger.info(
            "MCP server %r (%s) is at %s", target.name, target.kind.value, client.server_name
        )

    # Create the MCP function group
    group = DataRobotMCPFunctionGroup(config=config)

    # Store shared components for session client creation
    group._shared_auth_provider = auth_provider
    group._client_config = config

    # Set auth provider config defaults
    # For interactive OAuth2: use config values
    # For service accounts: default_user_id = server URL,
    #                       allow_default_user_id_for_tool_calls = True
    if auth_provider and target is not None:
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
        # `target` is None for a stdio server, which has no URL and no ref -- fall back
        # to the command, which is the only address such a server has.
        group.mcp_client_server_name = getattr(
            client, "server_name", target.url if target else config.server.command
        )
        group.mcp_client_transport = getattr(
            client, "transport", target.ref.transport if target else "stdio"
        )
        if not yielded:
            yield group
        else:
            # Cleanup (e.g. __aexit__) failed after we already yielded; re-raise so
            # the caller sees it and we do not yield a second time.
            raise
