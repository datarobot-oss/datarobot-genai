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
from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.dragent.plugins.datarobot_user_mcp_xaa_client import (
    get_mcp_auth_server_metadata_url,
)
from datarobot_genai.dragent.plugins.datarobot_user_mcp_xaa_client import get_xaa_params
from datarobot_genai.dragent.plugins.datarobot_user_mcp_xaa_client import get_xaa_params_from_config
from datarobot_genai.dragent.plugins.datarobot_user_mcp_xaa_client import (
    get_xaa_params_from_mcp_auth_server_metadata,
)
from datarobot_genai.dragent.plugins.datarobot_user_mcp_xaa_client import (
    mcp_client_with_xaa_support_function_group,
)
from datarobot_genai.dragent.plugins.datarobot_user_mcp_xaa_client import (
    parse_xaa_params_from_mcp_auth_server_metadata,
)
from datarobot_genai.dragent.plugins.datarobot_user_mcp_xaa_client import setup_auth_provider
from datarobot_genai.dragent.plugins.okta_a2a_auth import (
    OAuth2CrossApplicationAccessOAuth2AuthProvider,
)
from datarobot_genai.dragent.plugins.okta_a2a_auth import _CrossAppFlowParams


@pytest.fixture
def module_under_test() -> str:
    return "datarobot_genai.dragent.plugins.datarobot_user_mcp_xaa_client"


class TestMCPAuthServerMetadataDiscovery:
    @pytest.fixture
    def mock_async_http_client(self) -> Mock:
        mock_client = Mock()
        mock_client.get = AsyncMock(return_value=Mock())
        return mock_client

    @pytest.fixture
    def mock_get_retriable_async_http_client(
        self,
        module_under_test: str,
        mock_async_http_client: Mock,
    ) -> Iterator[Mock]:
        with patch(
            f"{module_under_test}.get_retriable_async_http_client",
        ) as mock_func:
            mock_func.return_value.__aenter__.return_value = mock_async_http_client
            yield mock_func

    @pytest.fixture
    def mock_get_mcp_auth_server_metadata_url(self, module_under_test: str) -> Iterator[Mock]:
        with patch(
            f"{module_under_test}.get_mcp_auth_server_metadata_url",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_parse_xaa_params_from_mcp_auth_server_metadata(
        self, module_under_test: str
    ) -> Iterator[Mock]:
        with patch(
            f"{module_under_test}.parse_xaa_params_from_mcp_auth_server_metadata",
        ) as mock_func:
            yield mock_func

    def test_parse_xaa_params_from_mcp_auth_server_metadata(self) -> None:
        mock_token_endpoint_auth_method = Mock()
        mock_trust_issuer = Mock()
        mock_exchange_audience = Mock()
        mock_token_url = Mock()
        mock_target_audience = Mock()
        mock_scopes = [Mock()]
        mcp_auth_server_metadata = {
            "urn:datarobot:nat_mcp_xaa_client": {
                "token_endpoint_auth_method": mock_token_endpoint_auth_method,
                "token_exchange": {
                    "trusted_issuer": mock_trust_issuer,
                    "audience": mock_exchange_audience,
                },
                "token_request": {
                    "token_url": mock_token_url,
                    "audience": mock_target_audience,
                    "scopes": mock_scopes,
                },
            }
        }
        output = parse_xaa_params_from_mcp_auth_server_metadata(mcp_auth_server_metadata)
        assert output == _CrossAppFlowParams(
            trusted_issuer=mock_trust_issuer,
            exchange_audience=mock_exchange_audience,
            token_url=mock_token_url,
            target_audience=mock_target_audience,
            id_jag_scopes=mock_scopes,
            token_endpoint_auth_method=mock_token_endpoint_auth_method,
        )

    def test_parse_xaa_params_from_mcp_auth_server_metadata_without_token_request_audience(
        self,
    ) -> None:
        mock_token_endpoint_auth_method = Mock()
        mock_trust_issuer = Mock()
        mock_exchange_audience = Mock()
        mock_token_url = Mock()
        mock_scopes = [Mock()]
        mcp_auth_server_metadata = {
            "urn:datarobot:nat_mcp_xaa_client": {
                "token_endpoint_auth_method": mock_token_endpoint_auth_method,
                "token_exchange": {
                    "trusted_issuer": mock_trust_issuer,
                    "audience": mock_exchange_audience,
                },
                "token_request": {
                    "token_url": mock_token_url,
                    "scopes": mock_scopes,
                },
            }
        }
        output = parse_xaa_params_from_mcp_auth_server_metadata(mcp_auth_server_metadata)
        assert output == _CrossAppFlowParams(
            trusted_issuer=mock_trust_issuer,
            exchange_audience=mock_exchange_audience,
            token_url=mock_token_url,
            target_audience=None,
            id_jag_scopes=mock_scopes,
            token_endpoint_auth_method=mock_token_endpoint_auth_method,
        )

    @pytest.mark.asyncio
    async def test_get_xaa_params_from_mcp_auth_server_metadata(
        self,
        mock_async_http_client: Mock,
        mock_get_mcp_auth_server_metadata_url: Mock,
        mock_get_retriable_async_http_client: Mock,
        mock_parse_xaa_params_from_mcp_auth_server_metadata: Mock,
    ) -> None:
        config = Mock()
        await get_xaa_params_from_mcp_auth_server_metadata(config)

        mock_get_mcp_auth_server_metadata_url.assert_called_once_with(config)
        mock_get_retriable_async_http_client.assert_called_once_with()
        mock_async_http_client.get.assert_called_once_with(
            mock_get_mcp_auth_server_metadata_url.return_value
        )
        mock_resp = mock_async_http_client.get.return_value
        mock_resp.raise_for_status.assert_called_once_with()
        mock_parse_xaa_params_from_mcp_auth_server_metadata.assert_called_once_with(
            mock_resp.json.return_value,
        )


class TestSetupAuthProvider:
    @pytest.fixture
    def mock_get_mcp_auth_server_metadata_url(self, module_under_test: str) -> Iterator[Mock]:
        with patch(
            f"{module_under_test}.get_mcp_auth_server_metadata_url",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_xaa_params_from_mcp_auth_server_metadata(
        self, module_under_test: str
    ) -> Iterator[AsyncMock]:
        with patch(
            f"{module_under_test}.get_xaa_params_from_mcp_auth_server_metadata",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_xaa_params_from_config(self, module_under_test: str) -> Iterator[AsyncMock]:
        with patch(f"{module_under_test}.get_xaa_params_from_config") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_xaa_params(self, module_under_test: str) -> Iterator[AsyncMock]:
        with patch(
            f"{module_under_test}.get_xaa_params",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    def test_get_mcp_auth_server_metadata_url(self) -> None:
        config = Mock()
        config.server.url = "https://foo:8081/bar/mcp"
        output = get_mcp_auth_server_metadata_url(config)
        assert output == "https://foo:8081/.well-known/oauth-protected-resource/bar/mcp"

    def test_get_xaa_params_from_config(self) -> None:
        xaa_config = Mock()
        output = get_xaa_params_from_config(xaa_config)

        assert output.trusted_issuer == xaa_config.token_exchange.trusted_issuer
        assert output.exchange_audience == xaa_config.token_exchange.audience
        assert output.token_url == xaa_config.token_request.token_url
        assert output.target_audience == xaa_config.token_request.audience
        assert output.id_jag_scopes == xaa_config.token_request.scopes
        assert output.token_endpoint_auth_method == xaa_config.token_endpoint_auth_method

    @pytest.mark.asyncio
    async def test_get_xaa_params_from_remote_mcp_auth_server_metadata(
        self,
        mock_get_xaa_params_from_config: Mock,
        mock_get_xaa_params_from_mcp_auth_server_metadata: AsyncMock,
    ) -> None:
        config = Mock()
        config.cross_application_access = None
        output = await get_xaa_params(config)

        mock_get_xaa_params_from_mcp_auth_server_metadata.assert_called_once_with(config)
        mock_get_xaa_params_from_config.assert_not_called()
        assert output == mock_get_xaa_params_from_mcp_auth_server_metadata.return_value

    @pytest.mark.asyncio
    async def test_get_xaa_params_from_local_config(
        self,
        mock_get_xaa_params_from_config: Mock,
        mock_get_xaa_params_from_mcp_auth_server_metadata: AsyncMock,
    ) -> None:
        config = Mock()
        output = await get_xaa_params(config)

        mock_get_xaa_params_from_config.assert_called_once_with(config.cross_application_access)
        mock_get_xaa_params_from_mcp_auth_server_metadata.assert_not_called()
        assert output == mock_get_xaa_params_from_config.return_value

    @pytest.mark.asyncio
    async def test_setup_auth_provider_with_inbound_headers_forwarded(
        self,
        mock_get_xaa_params: AsyncMock,
    ) -> None:
        mock_auth_provider = Mock()
        mock_config = Mock(forward_inbound_headers=True)
        _ = await setup_auth_provider(mock_auth_provider, mock_config)

        mock_get_xaa_params.assert_called_once_with(mock_config)
        mock_auth_provider.set_cross_app_flow_params.assert_called_once_with(
            mock_get_xaa_params.return_value,
        )
        mock_auth_provider.set_forward_inbound_x_datarobot_http_headers.assert_called_once_with(
            True
        )

    @pytest.mark.asyncio
    async def test_setup_auth_provider_without_inbound_headers_forwarded(
        self,
        mock_get_xaa_params: AsyncMock,
    ) -> None:
        mock_auth_provider = Mock()
        mock_config = Mock(forward_inbound_headers=False)
        _ = await setup_auth_provider(mock_auth_provider, mock_config)

        mock_get_xaa_params.assert_called_once_with(mock_config)
        mock_auth_provider.set_cross_app_flow_params.assert_called_once_with(
            mock_get_xaa_params.return_value,
        )
        mock_auth_provider.set_forward_inbound_http_headers.assert_not_called()


class TestSetupMCPClientWithXAASupportFunctionGroup:
    @pytest.fixture
    def mock_func_group(self) -> Mock:
        return Mock()

    @pytest.fixture
    def mock_per_user_mcp_client_function_group(
        self,
        mock_func_group: Mock,
        module_under_test: str,
    ) -> Iterator[Mock]:
        with patch(f"{module_under_test}.per_user_mcp_client_function_group") as mock_func:
            mock_func.return_value.__aenter__.return_value = mock_func_group
            yield mock_func

    @pytest.fixture
    def mock_setup_auth_provider(
        self,
        module_under_test: str,
    ) -> Iterator[AsyncMock]:
        with patch(
            f"{module_under_test}.setup_auth_provider",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_auth_provider(self) -> Mock:
        return Mock(spec=OAuth2CrossApplicationAccessOAuth2AuthProvider)

    @pytest.fixture
    def mock_nat_builder(self, mock_auth_provider: Mock) -> Mock:
        mock_builder = Mock()
        mock_builder.get_auth_provider = AsyncMock(return_value=mock_auth_provider)
        return mock_builder

    @pytest.mark.asyncio
    async def test_mcp_client_with_xaa_support_function_group(
        self,
        mock_auth_provider: Mock,
        mock_func_group: Mock,
        mock_nat_builder: AsyncMock,
        mock_per_user_mcp_client_function_group: Mock,
        mock_setup_auth_provider: AsyncMock,
    ) -> None:
        mock_config = Mock()
        async with mcp_client_with_xaa_support_function_group(
            mock_config,
            mock_nat_builder,
        ) as group:
            assert group == mock_func_group

        mock_nat_builder.get_auth_provider.assert_called_once_with(mock_config.server.auth_provider)
        mock_setup_auth_provider.assert_called_once_with(mock_auth_provider, mock_config)
        mock_per_user_mcp_client_function_group.assert_called_once_with(
            mock_config,
            mock_nat_builder,
        )

    @pytest.mark.asyncio
    async def test_mcp_client_with_xaa_support_function_group_error_with_incompatible_auth_provider(
        self,
        mock_auth_provider: Mock,
        mock_func_group: Mock,
        mock_nat_builder: AsyncMock,
        mock_per_user_mcp_client_function_group: Mock,
        mock_setup_auth_provider: AsyncMock,
    ) -> None:
        mock_nat_builder.get_auth_provider.return_value = Mock()
        mock_config = Mock()
        with pytest.raises(ValueError):
            async with mcp_client_with_xaa_support_function_group(
                mock_config,
                mock_nat_builder,
            ):
                pass

        mock_nat_builder.get_auth_provider.assert_called_once_with(mock_config.server.auth_provider)
        mock_setup_auth_provider.assert_not_called()
        mock_per_user_mcp_client_function_group.assert_not_called()
