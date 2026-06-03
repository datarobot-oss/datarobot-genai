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
from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import httpx
import pytest
from aiohttp import ClientResponseError
from fastmcp.server.transforms import Namespace

from datarobot_genai.drmcpbase.auth.exceptions import (
    NoDataRobotBearerTokenFoundInRequestContextError,
)
from datarobot_genai.drmcpbase.auth.exceptions import NoHeadersFoundInRequestContextError
from datarobot_genai.drmcpbase.drmcp_providers.user_mcp_provider import CACHE_TTL_IN_SECOND
from datarobot_genai.drmcpbase.drmcp_providers.user_mcp_provider import (
    HTTPX_CONNECT_POOL_WAIT_TIMEOUT_IN_SECOND,
)
from datarobot_genai.drmcpbase.drmcp_providers.user_mcp_provider import (
    MCP_K8S_POD_TCP_CONNECT_TIMEOUT_IN_SECOND,
)
from datarobot_genai.drmcpbase.drmcp_providers.user_mcp_provider import MCP_READ_TIMEOUT_IN_SECOND
from datarobot_genai.drmcpbase.drmcp_providers.user_mcp_provider import MCP_WRITE_TIMEOUT_IN_SECOND
from datarobot_genai.drmcpbase.drmcp_providers.user_mcp_provider import DataRobotClientWithAsyncAPI
from datarobot_genai.drmcpbase.drmcp_providers.user_mcp_provider import UserMCPProvider
from datarobot_genai.drmcpbase.drmcp_providers.user_mcp_provider import UserMCPProxyAuth
from datarobot_genai.drmcpbase.drmcp_providers.user_mcp_provider import get_user_mcp_endpoint
from datarobot_genai.drmcpbase.drmcp_providers.user_mcp_provider import httpx_async_client_factory
from datarobot_genai.drmcpbase.drmcp_providers.user_mcp_provider import (
    user_mcp_proxy_client_factory,
)


@pytest.fixture
def module_under_test() -> str:
    return "datarobot_genai.drmcpbase.drmcp_providers.user_mcp_provider"


@pytest.fixture
def mock_is_mcp_tools_gallery_support_enabled(module_under_test: str) -> Iterator[Mock]:
    with patch(
        f"{module_under_test}.is_mcp_tools_gallery_support_enabled_evaluated_with_existing_datarobot_client"
    ) as mock_func:
        yield mock_func


class TestUserMCPProvider:
    @pytest.fixture
    def mock_proxy_provider_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.ProxyProvider") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_datarobot_api_client(self) -> Mock:
        mock_client = Mock()
        mock_client._list_mcp_deployment_ids = AsyncMock()
        mock_client.clean_up = AsyncMock()
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

    @pytest.fixture
    def mock_user_mcp_proxy_client_factory(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.user_mcp_proxy_client_factory") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_datarobot_bearer_token_from_mcp_request_context(
        self, module_under_test: str
    ) -> Iterator[Mock]:
        with patch(
            f"{module_under_test}.get_datarobot_bearer_token_from_mcp_request_context"
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_api_v2_endpoint(self) -> Iterator[Mock]:
        with patch.object(DataRobotClientWithAsyncAPI, "get_api_v2_endpoint") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_asyncio_gather(self) -> Iterator[AsyncMock]:
        with patch.object(asyncio, "gather", new_callable=AsyncMock) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_proxy_provider(self) -> Mock:
        proxy_provider = Mock()
        proxy_provider.list_tools = AsyncMock()
        return proxy_provider

    @pytest.fixture
    def mock_get_or_create_mcp_proxy_provider(self) -> Iterator[Mock]:
        with patch.object(UserMCPProvider, "get_or_create_mcp_proxy_provider") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_user_mcp_proxy_providers_for_user(self) -> Iterator[Mock]:
        with patch.object(
            UserMCPProvider,
            "get_user_mcp_proxy_providers_for_user",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

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
        mock_datarobot_api_client.clean_up.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_private_method_list_tools(
        self,
        mock_asyncio_gather: AsyncMock,
        mock_get_datarobot_bearer_token_from_mcp_request_context: Mock,
        mock_get_user_mcp_proxy_providers_for_user: Mock,
        mock_is_mcp_tools_gallery_support_enabled: Mock,
        mock_proxy_provider: Mock,
    ) -> None:
        mock_get_user_mcp_proxy_providers_for_user.return_value = [mock_proxy_provider]

        mcp_provider = UserMCPProvider(Mock())
        await mcp_provider._list_tools()

        mock_get_datarobot_bearer_token_from_mcp_request_context.assert_called_once_with()
        mock_datarobot_api_token = (
            mock_get_datarobot_bearer_token_from_mcp_request_context.return_value
        )
        mock_is_mcp_tools_gallery_support_enabled.assert_called_once_with(
            mcp_provider.datarobot_api_client,
            mock_datarobot_api_token,
        )
        mock_get_user_mcp_proxy_providers_for_user.assert_called_once_with()
        mock_proxy_provider.list_tools.assert_called_once_with()
        mock_asyncio_gather.assert_called_once()
        assert mock_asyncio_gather.call_args.kwargs == {"return_exceptions": True}

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_get_user_mcp_proxy_providers_for_user",
        "mock_is_mcp_tools_gallery_support_enabled",
    )
    async def test_private_method_list_tools_ignore_exceptions_from_asyncio_gather(
        self,
        mock_asyncio_gather: AsyncMock,
    ) -> None:
        tools_in_one_proxy_provider = [Mock()]
        mock_asyncio_gather.return_value = [tools_in_one_proxy_provider, BaseException()]

        mcp_provider = UserMCPProvider(Mock())
        outputs = await mcp_provider._list_tools()

        mock_asyncio_gather.assert_called_once()
        assert mock_asyncio_gather.call_args.kwargs == {"return_exceptions": True}
        assert outputs == tools_in_one_proxy_provider

    def test_get_or_create_mcp_proxy_provider(
        self,
        mock_user_mcp_proxy_client_factory: Mock,
        mock_proxy_provider_cls: Mock,
    ) -> None:
        mcp_provider = UserMCPProvider(Mock())
        user_mcp_deployment_id = Mock()
        output = mcp_provider.get_or_create_mcp_proxy_provider(user_mcp_deployment_id)

        mock_user_mcp_proxy_client_factory.assert_called_once_with(
            mcp_provider.datarobot_api_endpoint,
            user_mcp_deployment_id,
        )
        mock_proxy_provider_cls.assert_called_once_with(
            mock_user_mcp_proxy_client_factory.return_value,
            CACHE_TTL_IN_SECOND,
        )
        mock_proxy_provider = mock_proxy_provider_cls.return_value
        (actual_namespace,), _ = mock_proxy_provider.wrap_transform.call_args
        assert isinstance(actual_namespace, Namespace)
        assert actual_namespace._prefix == user_mcp_deployment_id
        namespace_wrapped_proxy_provider = mock_proxy_provider.wrap_transform.return_value
        assert mcp_provider.user_mcp_proxy_providers == {
            user_mcp_deployment_id: namespace_wrapped_proxy_provider
        }
        assert output == namespace_wrapped_proxy_provider

    def test_get_or_create_mcp_proxy_provider_return_existing_one(
        self,
        mock_proxy_provider_cls: Mock,
    ) -> None:
        user_mcp_deployment_id = Mock()
        mcp_provider = UserMCPProvider(Mock())
        mock_existing_proxy_provider = Mock()
        mcp_provider.user_mcp_proxy_providers[user_mcp_deployment_id] = mock_existing_proxy_provider

        output = mcp_provider.get_or_create_mcp_proxy_provider(user_mcp_deployment_id)

        mock_proxy_provider_cls.assert_not_called()
        assert output == mock_existing_proxy_provider

    @pytest.mark.asyncio
    async def test_get_user_mcp_proxy_providers_for_user(
        self,
        mock_datarobot_api_client: Mock,
        mock_get_datarobot_bearer_token_from_mcp_request_context: Mock,
        mock_get_or_create_mcp_proxy_provider: Mock,
    ) -> None:
        user_mcp_deployment_id = Mock()
        mcp_provider = UserMCPProvider(Mock())
        mcp_provider.datarobot_api_client = mock_datarobot_api_client
        mock_datarobot_api_client._list_mcp_deployment_ids.return_value = [user_mcp_deployment_id]

        outputs = await mcp_provider.get_user_mcp_proxy_providers_for_user()

        mock_get_datarobot_bearer_token_from_mcp_request_context.assert_called_once_with()
        mock_datarobot_api_token = (
            mock_get_datarobot_bearer_token_from_mcp_request_context.return_value
        )
        mock_datarobot_api_client._list_mcp_deployment_ids.assert_called_once_with(
            mock_datarobot_api_token
        )
        mock_get_or_create_mcp_proxy_provider.assert_called_once_with(user_mcp_deployment_id)
        assert outputs == [mock_get_or_create_mcp_proxy_provider.return_value]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "side_effect",
        [
            NoHeadersFoundInRequestContextError,
            NoDataRobotBearerTokenFoundInRequestContextError,
            RuntimeError,
            ClientResponseError(Mock(), Mock()),
        ],
        ids=str,
    )
    async def test_get_user_mcp_proxy_providers_for_user_return_empty_if_errored(
        self,
        side_effect: NoHeadersFoundInRequestContextError
        | NoDataRobotBearerTokenFoundInRequestContextError
        | RuntimeError
        | ClientResponseError,
        mock_datarobot_api_client: Mock,
        mock_get_datarobot_bearer_token_from_mcp_request_context: Mock,
        mock_get_or_create_mcp_proxy_provider: Mock,
    ) -> None:
        mcp_provider = UserMCPProvider(Mock())
        mcp_provider.datarobot_api_client = mock_datarobot_api_client
        mock_datarobot_api_client._list_mcp_deployment_ids.side_effect = side_effect

        outputs = await mcp_provider.get_user_mcp_proxy_providers_for_user()

        mock_get_datarobot_bearer_token_from_mcp_request_context.assert_called_once_with()
        mock_datarobot_api_token = (
            mock_get_datarobot_bearer_token_from_mcp_request_context.return_value
        )
        mock_datarobot_api_client._list_mcp_deployment_ids.assert_called_once_with(
            mock_datarobot_api_token,
        )
        mock_get_or_create_mcp_proxy_provider.assert_not_called()
        assert outputs == []

    def test_get_user_mcp_endpoint(self, mock_get_api_v2_endpoint: Mock) -> None:
        mock_datarobot_endpoint = Mock()
        user_mcp_deployment_id = Mock()
        output = get_user_mcp_endpoint(mock_datarobot_endpoint, user_mcp_deployment_id)

        mock_get_api_v2_endpoint.assert_called_once_with(
            mock_datarobot_endpoint,
            f"deployments/{user_mcp_deployment_id}/directAccess/mcp",
        )
        assert output == mock_get_api_v2_endpoint.return_value


class TestMCPClientSetup:
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


class TestUserMCPAuth:
    @pytest.fixture
    def mock_fastmcp_get_http_request(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_http_request") as mock_func:
            yield mock_func

    @pytest.mark.asyncio
    async def test_update_outbound_request_header(
        self, mock_fastmcp_get_http_request: Mock
    ) -> None:
        auth_header_of_current_request = Mock()
        inbound_request = Mock(headers={"Authorization": auth_header_of_current_request})
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
