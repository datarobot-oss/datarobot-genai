# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# DataRobot, Inc. Confidential.
#
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
#
# The copyright notice above does not evidence any actual or intended
# publication of such source code.
import asyncio
import logging
from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from collections.abc import Callable
from collections.abc import Sequence
from contextlib import asynccontextmanager

import httpx
from aiohttp import ClientResponseError
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.providers import Provider
from fastmcp.server.providers.proxy import ProxyClient
from fastmcp.server.providers.proxy import ProxyProvider
from fastmcp.server.transforms import Namespace
from fastmcp.tools.base import Tool
from httpx import AsyncClient
from mcp.shared._httpx_utils import create_mcp_http_client

from datarobot_genai.drmcpbase.auth.exceptions import (
    NoDataRobotBearerTokenFoundInRequestContextError,
)
from datarobot_genai.drmcpbase.auth.exceptions import NoHeadersFoundInRequestContextError
from datarobot_genai.drmcpbase.auth.utils import get_datarobot_bearer_token_from_mcp_request_context
from datarobot_genai.drmcpbase.datarobot_services.client import DataRobotClientWithAsyncAPI
from datarobot_genai.drmcpbase.datarobot_services.client import TimeMeasurement
from datarobot_genai.drmcpbase.feature_flags import FeatureFlagEvaluation

logger = logging.getLogger(__name__)


CACHE_TTL_IN_SECOND = 10 * TimeMeasurement.MINUTE.to_numeric_value_in_second()
MCP_K8S_POD_TCP_CONNECT_TIMEOUT_IN_SECOND = 60 * TimeMeasurement.SECOND.to_numeric_value_in_second()
MCP_READ_TIMEOUT_IN_SECOND = 30 * TimeMeasurement.SECOND.to_numeric_value_in_second()
MCP_WRITE_TIMEOUT_IN_SECOND = 30 * TimeMeasurement.SECOND.to_numeric_value_in_second()
HTTPX_CONNECT_POOL_WAIT_TIMEOUT_IN_SECOND = 5 * TimeMeasurement.SECOND.to_numeric_value_in_second()


def get_user_mcp_endpoint(datarobot_public_api_endpoint: str, user_mcp_deployment_id: str) -> str:
    return DataRobotClientWithAsyncAPI.get_api_v2_endpoint(
        datarobot_public_api_endpoint, f"deployments/{user_mcp_deployment_id}/directAccess/mcp"
    )


class UserMCPProxyAuth(httpx.Auth):
    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        outbound_request = request
        try:
            inbound_request = get_http_request()
            auth_header_in_outbound_request = inbound_request.headers.get("Authorization")
            if auth_header_in_outbound_request:
                outbound_request.headers["Authorization"] = auth_header_in_outbound_request
        except RuntimeError as ex:
            message = (
                f"{str(ex)}. No authorization header (per user) is passed to "
                "outbound request against external user MCP."
            )
            logger.warning(message)

        yield outbound_request


def httpx_async_client_factory(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
    follow_redirects: bool = True,
) -> AsyncClient:
    # Both are intentionally ignored: custom K8s-tuned timeouts are always applied,
    # and follow_redirects=True is set internally by create_mcp_http_client.
    _ = timeout
    _ = follow_redirects
    return create_mcp_http_client(
        headers=headers,
        timeout=httpx.Timeout(
            connect=MCP_K8S_POD_TCP_CONNECT_TIMEOUT_IN_SECOND,
            read=MCP_READ_TIMEOUT_IN_SECOND,
            write=MCP_WRITE_TIMEOUT_IN_SECOND,
            pool=HTTPX_CONNECT_POOL_WAIT_TIMEOUT_IN_SECOND,
        ),
        auth=auth,
    )


def user_mcp_proxy_client_factory(
    datarobot_public_api_endpoint: str, user_mcp_deployment_id: str
) -> Callable[[], ProxyClient]:
    user_mcp_deployment_url = get_user_mcp_endpoint(
        datarobot_public_api_endpoint, user_mcp_deployment_id
    )

    def client_creation_factory() -> ProxyClient:
        return ProxyClient(
            StreamableHttpTransport(
                url=user_mcp_deployment_url,
                auth=UserMCPProxyAuth(),
                httpx_client_factory=httpx_async_client_factory,
            )
        )

    return client_creation_factory


class UserMCPProvider(Provider):
    def __init__(self, datarobot_api_endpoint: str) -> None:
        super().__init__()
        self.user_mcp_proxy_providers: dict[str, ProxyProvider] = {}
        self.datarobot_api_client: DataRobotClientWithAsyncAPI | None = None
        self.datarobot_api_endpoint = datarobot_api_endpoint

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        try:
            self.datarobot_api_client = DataRobotClientWithAsyncAPI(self.datarobot_api_endpoint)
            yield
        finally:
            await self.datarobot_api_client.clean_up()  # type: ignore[union-attr]

    def get_or_create_mcp_proxy_provider(self, user_mcp_deployment_id: str) -> ProxyProvider:
        if user_mcp_deployment_id in self.user_mcp_proxy_providers:
            return self.user_mcp_proxy_providers[user_mcp_deployment_id]
        user_mcp_proxy_provider = ProxyProvider(
            user_mcp_proxy_client_factory(self.datarobot_api_endpoint, user_mcp_deployment_id),
            CACHE_TTL_IN_SECOND,
        )
        self.user_mcp_proxy_providers[user_mcp_deployment_id] = (
            user_mcp_proxy_provider.wrap_transform(Namespace(user_mcp_deployment_id))  # type: ignore[assignment]
        )
        return self.user_mcp_proxy_providers[user_mcp_deployment_id]

    async def get_user_mcp_proxy_providers_for_user(self) -> Sequence[ProxyProvider]:
        try:
            datarobot_token = get_datarobot_bearer_token_from_mcp_request_context()
            user_mcp_deployment_ids = await self.datarobot_api_client._list_mcp_deployment_ids(  # type: ignore[union-attr]
                datarobot_token
            )
        except (
            NoHeadersFoundInRequestContextError,
            NoDataRobotBearerTokenFoundInRequestContextError,
            RuntimeError,
            ClientResponseError,
        ) as ex:
            logger.warning("Failed to list MCP deployments: %s. No user MCP provider returned.", ex)
            return []
        return [
            self.get_or_create_mcp_proxy_provider(user_mcp_deployment_id)
            for user_mcp_deployment_id in user_mcp_deployment_ids
        ]

    async def _list_tools(self) -> Sequence[Tool]:
        tools: list[Tool] = []
        if await FeatureFlagEvaluation.is_mcp_tools_gallery_support_enabled_for_user_in_mcp_request(
            self.datarobot_api_client,  # type: ignore[arg-type]
        ):
            results = await asyncio.gather(
                *[p.list_tools() for p in await self.get_user_mcp_proxy_providers_for_user()],
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, BaseException):
                    logger.warning("Failed to list tools from user MCP proxy: %s", r)
                else:
                    tools.extend(r)
        return tools
