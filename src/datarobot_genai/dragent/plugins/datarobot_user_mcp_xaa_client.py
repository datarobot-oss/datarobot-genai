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
import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from contextlib import AsyncExitStack
from typing import Any
from typing import Literal
from urllib.parse import urlsplit

import httpx
from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.cli.register_workflow import register_per_user_function_group
from nat.plugins.mcp.client.client_base import MCPBaseClient
from nat.plugins.mcp.client.client_config import MCPServerConfig
from nat.plugins.mcp.client.client_config import PerUserMCPClientConfig
from nat.plugins.mcp.client.client_impl import PerUserMCPFunctionGroup
from nat.plugins.mcp.client.client_impl import per_user_mcp_client_function_group
from pydantic import Field
from pydantic import HttpUrl

from datarobot_genai.dragent.cross_app_access_config import CrossApplicationAccessConfig
from datarobot_genai.dragent.cross_app_access_config import TokenEndpointAuthMethod
from datarobot_genai.dragent.http_client import get_retriable_async_http_client
from datarobot_genai.dragent.plugins.okta_a2a_auth import SESSION_INBOUND_HEADERS
from datarobot_genai.dragent.plugins.okta_a2a_auth import (
    OAuth2CrossApplicationAccessOAuth2AuthProvider,
)
from datarobot_genai.dragent.plugins.okta_a2a_auth import SessionInboundHeaders
from datarobot_genai.dragent.plugins.okta_a2a_auth import _CrossAppFlowParams

#: Member of the protected resource metadata document carrying the
#: Cross-Application Access block. Unlike the document's other non-RFC-9728
#: members it carries no ``x_`` prefix, so it matches this plugin's own
#: ``cross_application_access`` config field. Kept in sync with the field on
#: ``drmcpbase.oauth_protected_resource_metadata.entities``'s served metadata
#: (inlined rather than imported to avoid a dragent -> drmcpbase dependency).
CROSS_APPLICATION_ACCESS_METADATA_KEY = "cross_application_access"

logger = logging.getLogger(__name__)


def _requested_scopes(mcp_auth_server_metadata: dict[str, Any]) -> list[str]:
    """Return the scopes the exchange should ask for: ``scopes_supported``.

    That one list is already everything a usable token needs. The server
    publishes it as the union of what the exchange itself asks for on its
    second hop (``cross_application_access.token_request.scopes``, typically
    ``dr.impersonation`` — which no tool declares and no middleware reads) and
    what its own tools require; see
    ``drmcpbase.oauth_protected_resource_metadata.entities``.

    Asking for ``token_request.scopes`` alone was the bug: the agent arrived
    holding a token that opened no tool, because every scoped tool is hidden
    from ``tools/list`` and 403s on ``tools/call`` — which reads as a server
    exposing one tool rather than as a token short of a scope.

    Reading the union off the document rather than re-deriving it here is the
    point. RFC 9728 defines ``scopes_supported`` as the scopes "used in
    authorization requests to request access to this protected resource", so
    asking for exactly it is what the field is for; combining the two lists
    here would ask for a scope a server had deliberately withheld.

    What is granted stays the IdP's decision: Okta issues the intersection
    with the agent's resource connection, and a client not allowed a scope at
    all can be refused the whole exchange with ``invalid_scope``. Pin
    ``cross_application_access`` on this plugin's own config to ask for an
    exact list instead.
    """
    # Optional in the document: a server that enforces nothing and prescribes
    # nothing publishes no list at all, and the client then asks for no scopes
    # and lets the IdP decide.
    published = mcp_auth_server_metadata.get("scopes_supported") or []
    return [scope for scope in published if isinstance(scope, str) and scope.strip()]


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
        # The document's own `scopes_supported`, which is already the union of
        # what the exchange asks for and what the tools require. See
        # _requested_scopes.
        id_jag_scopes=_requested_scopes(mcp_auth_server_metadata),
        # Optional in the document; only private_key_jwt is implemented today.
        token_endpoint_auth_method=xaa_metadata.get(
            "token_endpoint_auth_method", TokenEndpointAuthMethod.PRIVATE_KEY_JWT.value
        ),
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
    url_path = url_split.path.rstrip("/")
    if url_path.endswith("/mcp"):
        url_path = url_path[: -len("/mcp")]
    return f"{url_split.scheme}://{url_split.netloc}{url_path}/.well-known/oauth-protected-resource"


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


def _wrap_session_calls(client: MCPBaseClient, session_headers: SessionInboundHeaders) -> None:
    """Send each call with its caller's token on a live session; fail it if the session dies."""

    async def connect_if_down() -> None:
        # NAT reconnects only inside a call on a live session. After its reconnect fails (the
        # server restarting), get_tool raises "not initialized" until something else revives it.
        if client.is_connected:
            return
        async with client._reconnect_lock:  # NAT's own: never tear down a session it just rebuilt
            if not client.is_connected:
                await client._run_lifecycle_command("reconnect")

    async def unless_session_dies(call: Awaitable[Any]) -> Any:
        # A call in flight when NAT's session task dies gets no answer and would wait out its
        # whole timeout (300 s). Fail it instead. No retry: the server may have run the tool.
        session_task = client._lifecycle_task
        call_task = asyncio.ensure_future(call)
        if session_task is None:
            return await call_task
        try:
            await asyncio.wait({call_task, session_task}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            call_task.cancel()
            raise
        if call_task.done():
            return call_task.result()
        call_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await call_task
        raise ConnectionError(f"MCP session to {client.server_name} closed during the call")

    # NAT's tool clients reach these through the client's attributes, so wrapping them here
    # covers every call. The real-client tests check this against NAT's code.
    for name in ("get_tool", "get_tools", "call_tool"):
        method = getattr(client, name)

        async def wrapped(*args: Any, _method: Any = method, **kwargs: Any) -> Any:
            session_headers.update(Context.get().metadata.headers)
            # A lifecycle task NAT starts again during this call copies this context.
            token = SESSION_INBOUND_HEADERS.set(session_headers)
            try:
                await connect_if_down()
                return await unless_session_dies(_method(*args, **kwargs))
            finally:
                SESSION_INBOUND_HEADERS.reset(token)

        setattr(client, name, wrapped)


@register_per_user_function_group(config_type=MCPClientWithXAASupportConfig)
async def mcp_client_with_xaa_support_function_group(
    config: MCPClientWithXAASupportConfig,
    builder: Builder,
) -> AsyncGenerator[PerUserMCPFunctionGroup, None]:
    auth_provider = await builder.get_auth_provider(config.server.auth_provider)
    if not isinstance(auth_provider, OAuth2CrossApplicationAccessOAuth2AuthProvider):
        raise ValueError("The auth_provider shall be a okta_cross_app_access type auth provider.")

    session_headers = SessionInboundHeaders(Context.get().metadata.headers)
    async with AsyncExitStack() as stack:
        try:
            await setup_auth_provider(auth_provider, config)
            # NAT starts the session's tasks while opening it, so they all see this holder.
            token = SESSION_INBOUND_HEADERS.set(session_headers)
            try:
                group = await stack.enter_async_context(
                    per_user_mcp_client_function_group(config, builder)
                )
            finally:
                SESSION_INBOUND_HEADERS.reset(token)
        except Exception:
            # The runtime enters this exception handling block when
            # - MCP well-known OAuth protected resource metadata retrieval fails, or
            #   the retrieved metadata is invalid
            # - MCP server connection fails (e.g., initial session connection: DNS,
            #   TCP, TLS)
            # - MCP list-tools transaction fails (even after retry) due to a
            #   connection or AuthN/AuthZ issue
            logger.error(
                "Failed to connect to MCP server at %s (connection or XAA exchange error). "
                "MCP tools from this server will be unavailable for this user session.",
                config.server.url,
                exc_info=True,
            )
            yield PerUserMCPFunctionGroup(config=config)
            return

        if group.mcp_client is not None:
            _wrap_session_calls(group.mcp_client, session_headers)
        yield group
