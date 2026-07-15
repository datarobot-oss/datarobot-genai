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
import logging
from collections.abc import AsyncGenerator
from typing import Any
from typing import Literal
from urllib.parse import urlsplit

import httpx
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_per_user_function_group
from nat.plugins.mcp.client.client_config import MCPServerConfig
from nat.plugins.mcp.client.client_config import PerUserMCPClientConfig
from nat.plugins.mcp.client.client_impl import PerUserMCPFunctionGroup
from nat.plugins.mcp.client.client_impl import per_user_mcp_client_function_group
from pydantic import Field
from pydantic import HttpUrl

from datarobot_genai.dragent.cross_app_access_config import CrossApplicationAccessConfig
from datarobot_genai.dragent.http_client import get_retriable_async_http_client
from datarobot_genai.dragent.plugins.okta_a2a_auth import (
    OAuth2CrossApplicationAccessOAuth2AuthProvider,
)
from datarobot_genai.dragent.plugins.okta_a2a_auth import _CrossAppFlowParams

logger = logging.getLogger(__name__)


def parse_xaa_params_from_mcp_auth_server_metadata(
    mcp_auth_server_metadata: dict[str, Any],
) -> _CrossAppFlowParams:
    xaa_metadata = mcp_auth_server_metadata["urn:datarobot:nat_mcp_xaa_client"]
    token_endpoint_auth_method = xaa_metadata["token_endpoint_auth_method"]
    token_exchange_metadata = xaa_metadata["token_exchange"]
    token_request_metadata = xaa_metadata["token_request"]

    return _CrossAppFlowParams(
        trusted_issuer=token_exchange_metadata["trusted_issuer"],
        exchange_audience=token_exchange_metadata["audience"],
        token_url=token_request_metadata["token_url"],
        target_audience=token_request_metadata.get("audience"),
        id_jag_scopes=token_request_metadata["scopes"],
        token_endpoint_auth_method=token_endpoint_auth_method,
    )


class CustomizedMCPServerConfig(MCPServerConfig):
    transport: Literal["streamable-http"] = Field(
        default="streamable-http",
        description=(
            "Transport type to connect to the MCP server (only streamable-http is supported)."
        ),
    )

    url: HttpUrl = Field(
        description="URL of the MCP server (for streamable-http transport).",
    )


class MCPClientWithXAASupportConfig(  # type: ignore[call-arg]
    PerUserMCPClientConfig,
    name="mcp_client_with_xaa_support",
):
    server: CustomizedMCPServerConfig = Field(
        description="Server connection details (transport, url/command, etc.)",
    )

    forward_inbound_headers: bool = Field(
        default=True,
        description=(
            "If set to True, all HTTP headers of inbound request are forwarded "
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
        except httpx.HTTPStatusError as exc:
            error_message = (
                "Failed to fetch MCP auth server metadata from "
                f"{mcp_auth_server_metadata_url}: {exc}"
            )
            logger.exception(error_message)
            raise exc

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
    try:
        return await get_xaa_params_from_mcp_auth_server_metadata(config)
    except httpx.HTTPStatusError:
        if config.cross_application_access:
            logger.info("Fall back to load XAA params from NAT workflow.yaml")
            return get_xaa_params_from_config(config.cross_application_access)
        raise RuntimeError(
            "Failed to load XAA params from both MCP well-known metadata and NAT workflow.yaml."
        )


async def setup_auth_provider(
    auth_provider: OAuth2CrossApplicationAccessOAuth2AuthProvider,
    config: MCPClientWithXAASupportConfig,
) -> OAuth2CrossApplicationAccessOAuth2AuthProvider:
    xaa_params = await get_xaa_params(config)
    auth_provider.set_cross_app_flow_params(xaa_params)

    if config.forward_inbound_headers:
        auth_provider.set_forward_inbound_http_headers(True)

    return auth_provider


@register_per_user_function_group(config_type=MCPClientWithXAASupportConfig)
async def mcp_client_with_xaa_support_function_group(
    config: MCPClientWithXAASupportConfig,
    builder: Builder,
) -> AsyncGenerator[PerUserMCPFunctionGroup, None]:
    auth_provider = await builder.get_auth_provider(config.server.auth_provider)
    if not isinstance(auth_provider, OAuth2CrossApplicationAccessOAuth2AuthProvider):
        raise ValueError("The auth_provider shall be a okta_cross_app_access type auth provider.")
    await setup_auth_provider(auth_provider, config)

    async with per_user_mcp_client_function_group(config, builder) as group:
        yield group
