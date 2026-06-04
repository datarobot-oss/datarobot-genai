# Copyright 2026 DataRobot, Inc.
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
import asyncio
import logging
from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from collections.abc import Callable
from collections.abc import Sequence
from contextlib import asynccontextmanager

import httpx
from aiohttp import ClientResponseError
from cachetools import LRUCache
from cachetools import cachedmethod
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.providers import Provider
from fastmcp.server.providers.proxy import ProxyClient
from fastmcp.server.providers.proxy import ProxyProvider
from fastmcp.server.transforms import Namespace
from fastmcp.tools.base import Tool
from httpx import AsyncClient
from mcp.shared._httpx_utils import create_mcp_http_client

from datarobot_genai.drmcpbase.auth.enums import DataRobotBearerHeaderEnum
from datarobot_genai.drmcpbase.auth.exceptions import InvalidBearerTokenError
from datarobot_genai.drmcpbase.auth.exceptions import (
    NoDataRobotBearerTokenFoundInRequestContextError,
)
from datarobot_genai.drmcpbase.auth.exceptions import NoHeadersFoundInRequestContextError
from datarobot_genai.drmcpbase.datarobot_services.client import DataRobotClientWithAsyncAPI
from datarobot_genai.drmcpbase.datarobot_services.client import TimeMeasurement
from datarobot_genai.drmcpbase.feature_flags import FeatureFlagEvaluation

logger = logging.getLogger(__name__)


CACHE_TTL_IN_SECOND = 10 * TimeMeasurement.MINUTE.to_numeric_value_in_second()
MCP_K8S_POD_TCP_CONNECT_TIMEOUT_IN_SECOND = 60 * TimeMeasurement.SECOND.to_numeric_value_in_second()
MCP_READ_TIMEOUT_IN_SECOND = 30 * TimeMeasurement.SECOND.to_numeric_value_in_second()
MCP_WRITE_TIMEOUT_IN_SECOND = 30 * TimeMeasurement.SECOND.to_numeric_value_in_second()
HTTPX_CONNECT_POOL_WAIT_TIMEOUT_IN_SECOND = 5 * TimeMeasurement.SECOND.to_numeric_value_in_second()
MAX_USER_MCP_TO_CACHE = 50


def get_user_mcp_endpoint(datarobot_public_api_endpoint: str, user_mcp_deployment_id: str) -> str:
    return f"{datarobot_public_api_endpoint}/deployments/{user_mcp_deployment_id}/directAccess/mcp"


class UserMCPProxyAuth(httpx.Auth):
    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        outbound_request = request
        try:
            inbound_request = get_http_request()
            auth_header_in_outbound_request = inbound_request.headers.get(
                DataRobotBearerHeaderEnum.AUTHORIZATION.get_normalized_header_key()
            )
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


class UserMCPProxyProviderCache:
    def __init__(self, datarobot_api_endpoint: str, max_cache_size: int = MAX_USER_MCP_TO_CACHE):
        self.datarobot_api_endpoint = datarobot_api_endpoint
        self._cache: LRUCache = LRUCache(maxsize=max_cache_size)

    @staticmethod
    def get_namespace_transform(user_mcp_deployment_id: str) -> Namespace:
        last_four_digit = user_mcp_deployment_id[-4:] or user_mcp_deployment_id
        return Namespace(f"user-mcp-{last_four_digit}")

    @cachedmethod(lambda self: self._cache)
    def get(self, user_mcp_deployment_id: str) -> ProxyProvider:
        user_mcp_proxy_provider = ProxyProvider(
            user_mcp_proxy_client_factory(self.datarobot_api_endpoint, user_mcp_deployment_id),
            CACHE_TTL_IN_SECOND,
        )
        return user_mcp_proxy_provider.wrap_transform(
            self.get_namespace_transform(user_mcp_deployment_id)
        )  # type: ignore[assignment]


class UserMCPProvider(Provider):
    def __init__(self, datarobot_api_endpoint: str) -> None:
        super().__init__()
        self.datarobot_api_client: DataRobotClientWithAsyncAPI | None = None
        self.datarobot_api_endpoint = datarobot_api_endpoint
        self.user_mcp_proxy_provider_cache = UserMCPProxyProviderCache(self.datarobot_api_endpoint)

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        async with DataRobotClientWithAsyncAPI(self.datarobot_api_endpoint) as client:
            self.datarobot_api_client = client
            yield

    def is_datarobot_api_client_initialized(self) -> bool:
        return self.datarobot_api_client is not None

    def get_or_create_mcp_proxy_provider(self, user_mcp_deployment_id: str) -> ProxyProvider:
        return self.user_mcp_proxy_provider_cache.get(user_mcp_deployment_id)

    async def get_user_mcp_deployment_ids(self) -> Sequence[str]:
        if not self.is_datarobot_api_client_initialized():
            logger.warning(
                "Failed to list MCP deployments. "
                "Because it executed it before MCP provider entered lifespan."
            )
            return []

        try:
            datarobot_token = DataRobotBearerHeaderEnum.AUTHORIZATION.get_from_mcp_request()
            return await self.datarobot_api_client._list_mcp_deployment_ids(  # type: ignore[union-attr]
                datarobot_token
            )
        except (
            NoHeadersFoundInRequestContextError,
            NoDataRobotBearerTokenFoundInRequestContextError,
            RuntimeError,
            ClientResponseError,
            InvalidBearerTokenError,
        ) as ex:
            logger.warning("Failed to list MCP deployments: %s.", ex)
            return []

    async def get_user_mcp_proxy_providers_for_user(self) -> Sequence[ProxyProvider]:
        return [
            self.get_or_create_mcp_proxy_provider(user_mcp_deployment_id)
            for user_mcp_deployment_id in await self.get_user_mcp_deployment_ids()
        ]

    async def _list_tools(self) -> Sequence[Tool]:
        tools: list[Tool] = []

        if (
            self.is_datarobot_api_client_initialized()
            and await FeatureFlagEvaluation.is_mcp_tools_gallery_support_enabled_for_user_in_mcp_request(  # noqa: E501
                self.datarobot_api_client,  # type: ignore[arg-type]
            )
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
