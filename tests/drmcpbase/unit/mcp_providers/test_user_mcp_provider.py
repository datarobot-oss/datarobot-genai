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
from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import httpx
import pytest
from aiohttp import ClientResponseError
from fastmcp.server.transforms import Namespace
from fastmcp.tools.base import Tool

from datarobot_genai.drmcpbase.auth.enums import DataRobotBearerHeaderEnum
from datarobot_genai.drmcpbase.auth.exceptions import (
    NoDataRobotBearerTokenFoundInRequestContextError,
)
from datarobot_genai.drmcpbase.auth.exceptions import NoHeadersFoundInRequestContextError
from datarobot_genai.drmcpbase.dynamic_tools.enums import DataRobotMCPToolCategory
from datarobot_genai.drmcpbase.feature_flags import FeatureFlagEvaluation
from datarobot_genai.drmcpbase.mcp_providers.user_mcp_provider import CACHE_TTL_IN_SECOND
from datarobot_genai.drmcpbase.mcp_providers.user_mcp_provider import (
    HTTPX_CONNECT_POOL_WAIT_TIMEOUT_IN_SECOND,
)
from datarobot_genai.drmcpbase.mcp_providers.user_mcp_provider import (
    MCP_K8S_POD_TCP_CONNECT_TIMEOUT_IN_SECOND,
)
from datarobot_genai.drmcpbase.mcp_providers.user_mcp_provider import MCP_READ_TIMEOUT_IN_SECOND
from datarobot_genai.drmcpbase.mcp_providers.user_mcp_provider import MCP_WRITE_TIMEOUT_IN_SECOND
from datarobot_genai.drmcpbase.mcp_providers.user_mcp_provider import UserMCPProvider
from datarobot_genai.drmcpbase.mcp_providers.user_mcp_provider import UserMCPProxyAuth
from datarobot_genai.drmcpbase.mcp_providers.user_mcp_provider import UserMCPProxyProviderCache
from datarobot_genai.drmcpbase.mcp_providers.user_mcp_provider import get_user_mcp_endpoint
from datarobot_genai.drmcpbase.mcp_providers.user_mcp_provider import httpx_async_client_factory
from datarobot_genai.drmcpbase.mcp_providers.user_mcp_provider import user_mcp_proxy_client_factory


@pytest.fixture
def module_under_test() -> str:
    return "datarobot_genai.drmcpbase.mcp_providers.user_mcp_provider"


@pytest.fixture
def mock_is_mcp_tools_gallery_support_enabled() -> Iterator[AsyncMock]:
    with patch.object(
        FeatureFlagEvaluation, "is_mcp_tools_gallery_support_enabled_for_user_in_mcp_request"
    ) as mock_func:
        yield mock_func


class TestUserMCPProvider:
    @pytest.fixture
    def mock_datarobot_api_client(self) -> Mock:
        mock_client = Mock()
        mock_client.__aenter__ = AsyncMock()
        mock_client.__aexit__ = AsyncMock()
        mock_client._list_mcp_deployment_ids = AsyncMock()
        return mock_client

    @pytest.fixture
    def mock_datarobot_api_client_cls(
        self,
        module_under_test: str,
        mock_datarobot_api_client: Mock,
    ) -> Iterator[Mock]:
        with patch(f"{module_under_test}.DataRobotClientWithAsyncAPI") as mock_cls:
            mock_cls.return_value = mock_datarobot_api_client
            yield mock_cls

    @pytest.fixture(autouse=True)
    def mock_user_mcp_proxy_provider_cache_cls(
        self,
        module_under_test: str,
    ) -> Iterator[Mock]:
        with patch(f"{module_under_test}.UserMCPProxyProviderCache") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_get_datarobot_bearer_token_from_x_datarobot_authorization(
        self,
        module_under_test: str,
    ) -> Iterator[Mock]:
        with patch.object(
            DataRobotBearerHeaderEnum.X_DATAROBOT_AUTHORIZATION,
            "get_from_mcp_request",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_asyncio_gather(self) -> Iterator[AsyncMock]:
        with patch.object(asyncio, "gather", new_callable=AsyncMock) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_mcp_proxy_provider(self) -> Mock:
        proxy_provider = Mock()
        proxy_provider.list_tools = AsyncMock()
        return proxy_provider

    @pytest.fixture
    def mock_get_or_create_mcp_proxy_provider(self) -> Iterator[Mock]:
        with patch.object(UserMCPProvider, "get_or_create_mcp_proxy_provider") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_user_mcp_deployment_ids(self) -> Iterator[AsyncMock]:
        with patch.object(UserMCPProvider, "get_user_mcp_deployment_ids") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_user_mcp_proxy_providers_for_user(self) -> Iterator[AsyncMock]:
        with patch.object(
            UserMCPProvider,
            "get_user_mcp_proxy_providers_for_user",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_is_datarobot_api_client_initialized(self) -> Iterator[Mock]:
        with patch.object(
            UserMCPProvider,
            "is_datarobot_api_client_initialized",
        ) as mock_func:
            yield mock_func

    def test_init(self, mock_user_mcp_proxy_provider_cache_cls: Mock) -> None:
        mock_datarobot_api_endpoint = Mock()
        mcp_provider = UserMCPProvider(mock_datarobot_api_endpoint)

        mock_user_mcp_proxy_provider_cache_cls.assert_called_once_with(mock_datarobot_api_endpoint)
        assert (
            mcp_provider.user_mcp_proxy_provider_cache
            == mock_user_mcp_proxy_provider_cache_cls.return_value
        )

    @pytest.mark.asyncio
    async def test_datarobot_api_client_init_and_clean_up(
        self,
        mock_datarobot_api_client_cls: Mock,
        mock_datarobot_api_client: Mock,
    ) -> None:
        mock_datarobot_api_endpoint = Mock()
        mcp_provider = UserMCPProvider(mock_datarobot_api_endpoint)

        async with mcp_provider.lifespan():
            pass

        mock_datarobot_api_client_cls.assert_called_once_with(mock_datarobot_api_endpoint)
        assert (
            mcp_provider.datarobot_api_client == mock_datarobot_api_client.__aenter__.return_value
        )
        mock_datarobot_api_client.__aexit__.assert_called_once_with(None, None, None)

    @pytest.mark.asyncio
    async def test_private_method_list_tools(
        self,
        mock_asyncio_gather: AsyncMock,
        mock_get_user_mcp_proxy_providers_for_user: AsyncMock,
        mock_is_datarobot_api_client_initialized: Mock,
        mock_is_mcp_tools_gallery_support_enabled: Mock,
        mock_mcp_proxy_provider: Mock,
    ) -> None:
        mock_get_user_mcp_proxy_providers_for_user.return_value = [mock_mcp_proxy_provider]
        mock_is_datarobot_api_client_initialized.return_value = True

        mcp_provider = UserMCPProvider(Mock())
        mcp_provider.datarobot_api_client = Mock()
        await mcp_provider._list_tools()

        mock_is_mcp_tools_gallery_support_enabled.assert_called_once_with(
            mcp_provider.datarobot_api_client,
        )
        mock_get_user_mcp_proxy_providers_for_user.assert_called_once_with()
        mock_mcp_proxy_provider.list_tools.assert_called_once_with()
        mock_asyncio_gather.assert_called_once()
        assert mock_asyncio_gather.call_args.kwargs == {"return_exceptions": True}

    @pytest.mark.asyncio
    async def test_private_method_list_tools_return_empty_if_dr_api_client_uninitialized(
        self,
        mock_asyncio_gather: AsyncMock,
        mock_get_user_mcp_proxy_providers_for_user: AsyncMock,
        mock_is_datarobot_api_client_initialized: Mock,
        mock_is_mcp_tools_gallery_support_enabled: Mock,
        mock_mcp_proxy_provider: Mock,
    ) -> None:
        mock_is_datarobot_api_client_initialized.return_value = False

        mcp_provider = UserMCPProvider(Mock())

        output = await mcp_provider._list_tools()

        mock_is_mcp_tools_gallery_support_enabled.assert_not_called()
        mock_get_user_mcp_proxy_providers_for_user.assert_not_called()
        mock_mcp_proxy_provider.list_tools.assert_not_called()
        mock_asyncio_gather.assert_not_called()
        assert output == []

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_get_user_mcp_proxy_providers_for_user",
        "mock_is_mcp_tools_gallery_support_enabled",
    )
    async def test_private_method_list_tools_ignore_exceptions_from_asyncio_gather(
        self,
        mock_asyncio_gather: AsyncMock,
        mock_is_datarobot_api_client_initialized: Mock,
    ) -> None:
        def remote_search(q: str) -> str:
            """Search."""
            return q

        proxied_tool = Tool.from_function(fn=remote_search, name="user-mcp-ab12_search")
        mock_asyncio_gather.return_value = [[proxied_tool], BaseException()]
        mock_is_datarobot_api_client_initialized.return_value = True

        mcp_provider = UserMCPProvider(Mock())
        mcp_provider.datarobot_api_client = Mock()
        outputs = await mcp_provider._list_tools()

        mock_asyncio_gather.assert_called_once()
        assert mock_asyncio_gather.call_args.kwargs == {"return_exceptions": True}
        # The healthy proxy's tool is returned, stamped so the gallery sees it as a
        # hosted, third-party proxied user-MCP tool; the failed proxy is dropped.
        assert len(outputs) == 1
        assert outputs[0].name == "user-mcp-ab12_search"
        assert outputs[0].meta == {"tool_category": DataRobotMCPToolCategory.PROXIED_USER_MCP.name}

    def test_get_or_create_mcp_proxy_provider(
        self,
        mock_user_mcp_proxy_provider_cache_cls: Mock,
    ) -> None:
        mcp_provider = UserMCPProvider(Mock())
        user_mcp_deployment_id = Mock()
        output = mcp_provider.get_or_create_mcp_proxy_provider(user_mcp_deployment_id)

        mcp_provider.user_mcp_proxy_provider_cache.get.assert_called_once_with(
            user_mcp_deployment_id
        )
        assert output == mcp_provider.user_mcp_proxy_provider_cache.get.return_value

    @pytest.mark.asyncio
    async def test_get_user_mcp_deployment_ids(
        self,
        mock_datarobot_api_client: Mock,
        mock_get_datarobot_bearer_token_from_x_datarobot_authorization: Mock,
    ) -> None:
        user_mcp_deployment_id = Mock()
        mcp_provider = UserMCPProvider(Mock())
        mcp_provider.datarobot_api_client = mock_datarobot_api_client
        mock_datarobot_api_client._list_mcp_deployment_ids.return_value = [user_mcp_deployment_id]

        outputs = await mcp_provider.get_user_mcp_deployment_ids()

        mock_get_datarobot_bearer_token_from_x_datarobot_authorization.assert_called_once_with()
        mock_datarobot_api_token = (
            mock_get_datarobot_bearer_token_from_x_datarobot_authorization.return_value
        )
        mock_datarobot_api_client._list_mcp_deployment_ids.assert_called_once_with(
            mock_datarobot_api_token
        )
        assert outputs == [user_mcp_deployment_id]

    @pytest.mark.asyncio
    async def test_get_user_mcp_deployment_ids_return_empty_with_uninitialized_dr_api_client(
        self,
        mock_datarobot_api_client: Mock,
        mock_get_datarobot_bearer_token_from_x_datarobot_authorization: Mock,
    ) -> None:
        mcp_provider = UserMCPProvider(Mock())
        mcp_provider.datarobot_api_client = None

        outputs = await mcp_provider.get_user_mcp_deployment_ids()

        mock_datarobot_api_client._list_mcp_deployment_ids.assert_not_called()
        assert outputs == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raised_error",
        [
            NoHeadersFoundInRequestContextError,
            NoDataRobotBearerTokenFoundInRequestContextError,
            RuntimeError,
            ClientResponseError(Mock(), Mock()),
        ],
        ids=str,
    )
    async def test_get_user_mcp_deployment_ids_return_empty_if_errored(
        self,
        raised_error: NoHeadersFoundInRequestContextError
        | NoDataRobotBearerTokenFoundInRequestContextError
        | RuntimeError
        | ClientResponseError,
        mock_datarobot_api_client: Mock,
        mock_get_datarobot_bearer_token_from_x_datarobot_authorization: Mock,
    ) -> None:
        mcp_provider = UserMCPProvider(Mock())
        mcp_provider.datarobot_api_client = mock_datarobot_api_client
        mock_datarobot_api_client._list_mcp_deployment_ids.side_effect = raised_error

        outputs = await mcp_provider.get_user_mcp_deployment_ids()

        mock_get_datarobot_bearer_token_from_x_datarobot_authorization.assert_called_once_with()
        mock_datarobot_api_token = (
            mock_get_datarobot_bearer_token_from_x_datarobot_authorization.return_value
        )
        mock_datarobot_api_client._list_mcp_deployment_ids.assert_called_once_with(
            mock_datarobot_api_token,
        )
        assert outputs == []

    @pytest.mark.asyncio
    async def test_get_user_mcp_proxy_providers_for_user(
        self,
        mock_get_or_create_mcp_proxy_provider: Mock,
        mock_get_user_mcp_deployment_ids: AsyncMock,
    ) -> None:
        user_mcp_deployment_id = Mock()
        mock_get_user_mcp_deployment_ids.return_value = [user_mcp_deployment_id]

        mcp_provider = UserMCPProvider(Mock())
        outputs = await mcp_provider.get_user_mcp_proxy_providers_for_user()

        mock_get_user_mcp_deployment_ids.assert_called_once_with()
        mock_get_or_create_mcp_proxy_provider.assert_called_once_with(user_mcp_deployment_id)
        assert outputs == [mock_get_or_create_mcp_proxy_provider.return_value]

    def test_get_user_mcp_endpoint(self) -> None:
        mock_datarobot_endpoint = Mock()
        user_mcp_deployment_id = Mock()
        output = get_user_mcp_endpoint(mock_datarobot_endpoint, user_mcp_deployment_id)

        assert output == (
            f"{mock_datarobot_endpoint}/deployments/{user_mcp_deployment_id}/directAccess/mcp"
        )


class TestUserMCPProxyProviderCache:
    @pytest.fixture
    def mock_user_mcp_proxy_client_factory(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.user_mcp_proxy_client_factory") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_proxy_provider_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.ProxyProvider") as mock_cls:
            yield mock_cls

    def test_get(
        self,
        mock_user_mcp_proxy_client_factory: Mock,
        mock_proxy_provider_cls: Mock,
    ) -> None:
        datarobot_api_endpoint = Mock()
        provider_cache = UserMCPProxyProviderCache(datarobot_api_endpoint, 1)

        user_mcp_deployment_id = "dafadfsa"
        output = provider_cache.get(user_mcp_deployment_id)

        mock_user_mcp_proxy_client_factory.assert_called_once_with(
            datarobot_api_endpoint,
            user_mcp_deployment_id,
        )
        mock_proxy_provider_cls.assert_called_once_with(
            mock_user_mcp_proxy_client_factory.return_value,
            CACHE_TTL_IN_SECOND,
        )
        mock_proxy_provider = mock_proxy_provider_cls.return_value
        (actual_namespace,), _ = mock_proxy_provider.wrap_transform.call_args
        assert isinstance(actual_namespace, Namespace)
        assert output == mock_proxy_provider.wrap_transform.return_value

    def test_get_returns_cached_result(
        self,
        mock_user_mcp_proxy_client_factory: Mock,
        mock_proxy_provider_cls: Mock,
    ) -> None:
        provider_cache = UserMCPProxyProviderCache(Mock(), 1)
        user_mcp_deployment_id = "dafadfsa"
        first_result = provider_cache.get(user_mcp_deployment_id)
        second_result = provider_cache.get(user_mcp_deployment_id)

        mock_proxy_provider_cls.assert_called_once_with(
            mock_user_mcp_proxy_client_factory.return_value,
            CACHE_TTL_IN_SECOND,
        )
        assert first_result == second_result

    @pytest.mark.parametrize(
        "user_mcp_deployment_id, namespace_transform_value",
        [
            ("dsafdafd", "dafd"),
            ("fd", "fd"),
        ],
    )
    def test_get_namespace_transform(
        self,
        user_mcp_deployment_id: str,
        namespace_transform_value: str,
    ) -> None:
        output = UserMCPProxyProviderCache.get_namespace_transform(user_mcp_deployment_id)
        assert output._prefix == f"user-mcp-{namespace_transform_value}"


class TestMCPProxyClientSetup:
    @pytest.fixture
    def mock_get_user_mcp_endpoint(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_user_mcp_endpoint") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_proxy_client_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.ProxyClient") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_streamable_http_transport_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.StreamableHttpTransport") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_user_mcp_proxy_auth_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.UserMCPProxyAuth") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_async_client_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.AsyncClient") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_httpx_async_client_factory(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.httpx_async_client_factory") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_create_mcp_http_client(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.create_mcp_http_client") as mock_func:
            yield mock_func

    def test_httpx_async_client_factory(
        self,
        mock_create_mcp_http_client: Mock,
    ) -> None:
        mock_headers = Mock()
        mock_auth = Mock()
        output = httpx_async_client_factory(mock_headers, Mock(), mock_auth, Mock())

        mock_create_mcp_http_client.assert_called_once_with(
            headers=mock_headers,
            timeout=httpx.Timeout(
                connect=MCP_K8S_POD_TCP_CONNECT_TIMEOUT_IN_SECOND,
                read=MCP_READ_TIMEOUT_IN_SECOND,
                write=MCP_WRITE_TIMEOUT_IN_SECOND,
                pool=HTTPX_CONNECT_POOL_WAIT_TIMEOUT_IN_SECOND,
            ),
            auth=mock_auth,
        )
        assert output == mock_create_mcp_http_client.return_value

    def test_get_user_mcp_proxy_client_factory(
        self,
        mock_get_user_mcp_endpoint: Mock,
        mock_httpx_async_client_factory: Mock,
        mock_proxy_client_cls: Mock,
        mock_streamable_http_transport_cls: Mock,
        mock_user_mcp_proxy_auth_cls: Mock,
    ) -> None:
        datarobot_public_api_endpoint = Mock()
        user_mcp_deployment_id = Mock()
        factory_func = user_mcp_proxy_client_factory(
            datarobot_public_api_endpoint, user_mcp_deployment_id
        )
        output = factory_func()

        mock_get_user_mcp_endpoint.assert_called_once_with(
            datarobot_public_api_endpoint, user_mcp_deployment_id
        )
        mock_user_mcp_proxy_auth_cls.assert_called_once_with()
        mock_streamable_http_transport_cls.assert_called_once_with(
            url=mock_get_user_mcp_endpoint.return_value,
            auth=mock_user_mcp_proxy_auth_cls.return_value,
            httpx_client_factory=mock_httpx_async_client_factory,
        )
        mock_proxy_client_cls.assert_called_once_with(
            mock_streamable_http_transport_cls.return_value
        )
        assert output == mock_proxy_client_cls.return_value


class TestUserMCPProxyAuth:
    @pytest.fixture
    def mock_fastmcp_get_http_request(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_http_request") as mock_func:
            yield mock_func

    @pytest.mark.asyncio
    async def test_update_outbound_request_header(
        self, mock_fastmcp_get_http_request: Mock
    ) -> None:
        auth_header_of_current_request = Mock()
        inbound_request = Mock(
            headers={
                DataRobotBearerHeaderEnum.X_DATAROBOT_AUTHORIZATION.get_normalized_header_key(): auth_header_of_current_request,  # noqa: E501
            }
        )
        mock_fastmcp_get_http_request.return_value = inbound_request
        auth = UserMCPProxyAuth()

        outbound_request = Mock(headers={})
        async for _ in auth.async_auth_flow(outbound_request):
            pass

        mock_fastmcp_get_http_request.assert_called_once_with()
        assert outbound_request.headers["Authorization"] == auth_header_of_current_request

    @pytest.mark.asyncio
    async def test_not_update_outbound_request_header_if_no_bearer_auth_in_inbound_request(
        self,
        mock_fastmcp_get_http_request: Mock,
    ) -> None:
        inbound_request = Mock(headers={})
        mock_fastmcp_get_http_request.return_value = inbound_request
        auth = UserMCPProxyAuth()

        original_outbound_request_headers = Mock()
        outbound_request = Mock(headers=original_outbound_request_headers)
        async for _ in auth.async_auth_flow(outbound_request):
            pass
        mock_fastmcp_get_http_request.assert_called_once_with()
        assert outbound_request.headers == original_outbound_request_headers

    @pytest.mark.asyncio
    async def test_not_update_outbound_request_header_if_no_active_request_found(
        self,
        mock_fastmcp_get_http_request: Mock,
    ) -> None:
        mock_fastmcp_get_http_request.side_effect = RuntimeError()
        auth = UserMCPProxyAuth()

        original_user_mcp_request_headers = Mock()
        user_mcp_request = Mock(headers=original_user_mcp_request_headers)
        async for _ in auth.async_auth_flow(user_mcp_request):
            pass
        mock_fastmcp_get_http_request.assert_called_once_with()
        assert user_mcp_request.headers == original_user_mcp_request_headers
