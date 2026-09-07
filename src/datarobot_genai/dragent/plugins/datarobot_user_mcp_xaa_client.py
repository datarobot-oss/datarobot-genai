# Copyright 2026 DataRobot, Inc. and its affiliates.
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
from collections.abc import AsyncGenerator
from typing import Any
from typing import Literal
from urllib.parse import urlsplit

import httpx
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_per_user_function_group
from nat.data_models.component_ref import AuthenticationRef
from nat.plugins.mcp.client.client_config import PerUserMCPClientConfig
from nat.plugins.mcp.client.client_impl import PerUserMCPFunctionGroup
from nat.plugins.mcp.client.client_impl import per_user_mcp_client_function_group
from pydantic import Field
from pydantic import HttpUrl

from datarobot_genai.core.mcp.target import build_target
from datarobot_genai.dragent.cross_app_access_config import CrossApplicationAccessConfig
from datarobot_genai.dragent.cross_app_access_config import TokenEndpointAuthMethod
from datarobot_genai.dragent.http_client import get_retriable_async_http_client
from datarobot_genai.dragent.plugins.datarobot_mcp_client import DataRobotMCPServerConfig
from datarobot_genai.dragent.plugins.datarobot_mcp_client import resolve_auth_provider_name
from datarobot_genai.dragent.plugins.datarobot_mcp_client import resolve_server_ref
from datarobot_genai.dragent.plugins.okta_a2a_auth import (
    OAuth2CrossApplicationAccessOAuth2AuthProvider,
)
from datarobot_genai.dragent.plugins.okta_a2a_auth import _CrossAppFlowParams

#: Member of the protected resource metadata document carrying the
#: Cross-Application Access block. Unlike the document's other non-RFC-9728
#: members it carries no ``x_`` prefix, so it matches this plugin's own
#: ``cross_application_access`` config field. Kept in sync with the field on
#: ``drmcpbase.oauth_protected_resource_metadata.entities``'s served metadata
#: (inlined rather than imported to avoid a dragent -> drmcpbase dependency).
CROSS_APPLICATION_ACCESS_METADATA_KEY = "cross_application_access"


def parse_xaa_params_from_mcp_auth_server_metadata(
    mcp_auth_server_metadata: dict[str, Any],
) -> _CrossAppFlowParams:
    xaa_metadata = mcp_auth_server_metadata.get(CROSS_APPLICATION_ACCESS_METADATA_KEY)
    if not xaa_metadata:
        raise RuntimeError(
            "MCP auth server metadata declares no "
            f"`{CROSS_APPLICATION_ACCESS_METADATA_KEY}` block. Either configure "
            "`cross_application_access` on the MCP client, or publish it from the "
            "MCP server's MCP_XAA_* settings."
        )

    missing = [key for key in ("token_exchange", "token_request") if key not in xaa_metadata]
    if missing:
        raise RuntimeError(
            f"MCP auth server metadata `{CROSS_APPLICATION_ACCESS_METADATA_KEY}` block "
            f"is missing required {', '.join(missing)}."
        )

    token_exchange_metadata = xaa_metadata["token_exchange"]
    token_request_metadata = xaa_metadata["token_request"]

    return _CrossAppFlowParams(
        trusted_issuer=token_exchange_metadata["trusted_issuer"],
        exchange_audience=token_exchange_metadata["audience"],
        token_url=token_request_metadata["token_url"],
        target_audience=token_request_metadata.get("audience"),
        id_jag_scopes=token_request_metadata["scopes"],
        # Optional in the document; only private_key_jwt is implemented today.
        token_endpoint_auth_method=xaa_metadata.get(
            "token_endpoint_auth_method", TokenEndpointAuthMethod.PRIVATE_KEY_JWT.value
        ),
    )


class CustomizedMCPServerConfig(DataRobotMCPServerConfig):
    """The XAA client's server block: the same addressing as ``datarobot_mcp_client``.

    Subclasses it rather than NAT's base so the two client types cannot drift. That is
    what gives this one ``name``, and with it an address that can live in the
    environment as ``<name>_mcp_*`` instead of being pinned in the YAML -- so an XAA
    server can differ between staging and production without editing the workflow.

    Narrowed in two ways. ``transport`` is ``streamable-http`` only, which is all the
    per-user MCP path supports, and ``auth_provider`` has no default: this client
    requires an ``okta_cross_app_access`` provider, and inheriting a default of
    ``datarobot_mcp_auth`` would mean fetching the wrong provider and failing an
    isinstance check rather than saying what is missing.
    """

    transport: Literal["streamable-http"] = Field(  # type: ignore[assignment]
        default="streamable-http",
        description=(
            "Transport type to connect to the MCP server (only streamable-http is supported)."
        ),
    )

    url: HttpUrl | None = Field(  # type: ignore[assignment]
        default=None,
        description=(
            "URL of the MCP server. Optional: leave it unset and give `name` instead, "
            "and the address is resolved from <name>_mcp_* in the environment."
        ),
    )

    auth_provider: str | AuthenticationRef | None = Field(
        default=None,
        description="Reference to an `okta_cross_app_access` authentication provider.",
    )


class MCPClientWithXAASupportConfig(  # type: ignore[call-arg]
    PerUserMCPClientConfig,
    name="mcp_client_with_xaa_support",
):
    server: CustomizedMCPServerConfig = Field(
        description="Server connection details (transport, url/command, etc.)",
    )

    forward_inbound_headers: bool = Field(
        default=False,
        description=(
            "If set to True, selected x-datarobot-* HTTP headers of inbound request are forwarded "
            "except for reserved headers configured in auth_provider."
        ),
    )

    cross_application_access: CrossApplicationAccessConfig | None = Field(
        default=None,
        description=(
            "Configuration for Cross-Application Access utilizing a hybrid RFC 8693 / "
            "RFC 7523 flow. If not configured, it will be read from MCP auth server metadata."
        ),
    )


def get_mcp_auth_server_metadata_url(
    config: MCPClientWithXAASupportConfig,
) -> str:
    mcp_server_url = str(config.server.url)
    url_split = urlsplit(mcp_server_url)
    return (
        f"{url_split.scheme}://{url_split.netloc}"
        f"/.well-known/oauth-protected-resource{url_split.path}"
    )


async def get_xaa_params_from_mcp_auth_server_metadata(
    config: MCPClientWithXAASupportConfig,
) -> _CrossAppFlowParams:
    mcp_auth_server_metadata_url = get_mcp_auth_server_metadata_url(config)
    async with get_retriable_async_http_client() as http_client:
        try:
            resp = await http_client.get(mcp_auth_server_metadata_url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Failed to fetch MCP auth server metadata from "
                f"{mcp_auth_server_metadata_url}: {exc}"
            )

    return parse_xaa_params_from_mcp_auth_server_metadata(resp.json())


def get_xaa_params_from_config(xaa_config: CrossApplicationAccessConfig) -> _CrossAppFlowParams:
    return _CrossAppFlowParams(
        trusted_issuer=xaa_config.token_exchange.trusted_issuer,
        exchange_audience=xaa_config.token_exchange.audience,
        token_url=xaa_config.token_request.token_url,
        target_audience=xaa_config.token_request.audience,
        id_jag_scopes=xaa_config.token_request.scopes,
        token_endpoint_auth_method=xaa_config.token_endpoint_auth_method,
    )


async def get_xaa_params(config: MCPClientWithXAASupportConfig) -> _CrossAppFlowParams:
    if config.cross_application_access:
        return get_xaa_params_from_config(config.cross_application_access)
    return await get_xaa_params_from_mcp_auth_server_metadata(config)


async def setup_auth_provider(
    auth_provider: OAuth2CrossApplicationAccessOAuth2AuthProvider,
    config: MCPClientWithXAASupportConfig,
) -> OAuth2CrossApplicationAccessOAuth2AuthProvider:
    xaa_params = await get_xaa_params(config)
    auth_provider.set_cross_app_flow_params(xaa_params)

    if config.forward_inbound_headers:
        auth_provider.set_forward_inbound_x_datarobot_http_headers(True)

    return auth_provider


@register_per_user_function_group(config_type=MCPClientWithXAASupportConfig)
async def mcp_client_with_xaa_support_function_group(
    config: MCPClientWithXAASupportConfig,
    builder: Builder,
) -> AsyncGenerator[PerUserMCPFunctionGroup, None]:
    from datarobot_genai.core.config import resolve_config  # noqa: PLC0415

    if not config.server.auth_provider:
        raise ValueError(
            f"MCP block {config.server.name!r} uses cross-application access but names no "
            f"`auth_provider`. Declare an `okta_cross_app_access` entry under "
            f"`authentication:` and reference it here."
        )

    # Resolve the address exactly as `datarobot_mcp_client` does, so `name` plus
    # <name>_mcp_* in the environment works here too, and write it back onto the block:
    # NAT's per-user client reads `config.server.url`, and so does the
    # protected-resource metadata lookup inside `setup_auth_provider` below.
    #
    # The block's `auth_provider` is applied to the ref first. That is what tells
    # `build_target` this server presents an exchanged per-user token rather than the
    # DataRobot service one, so it does not demand a DATAROBOT_API_TOKEN that an XAA
    # deployment typically does not have.
    if config.server.url is None:
        app_config = resolve_config()
        ref = resolve_server_ref(config.server, app_config)
        ref = ref.model_copy(
            update={"auth_provider": str(resolve_auth_provider_name(config.server, ref))}
        )
        config.server.url = HttpUrl(
            build_target(
                ref,
                datarobot_endpoint=app_config.resolve_datarobot_endpoint(),
                datarobot_api_token=app_config.resolve_datarobot_api_token(),
            ).url
        )

    auth_provider = await builder.get_auth_provider(config.server.auth_provider)
    if not isinstance(auth_provider, OAuth2CrossApplicationAccessOAuth2AuthProvider):
        raise ValueError("The auth_provider shall be a okta_cross_app_access type auth provider.")
    await setup_auth_provider(auth_provider, config)

    async with per_user_mcp_client_function_group(config, builder) as group:
        yield group
