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
from nat.builder.context import Context
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred
from nat.data_models.authentication import HeaderCred
from pydantic import SecretStr

from datarobot_genai.dragent.plugins.mcp_xaa_auth import MCPXAAAuthProvider
from datarobot_genai.dragent.plugins.mcp_xaa_auth import MCPXAAAuthProviderConfig
from datarobot_genai.dragent.plugins.mcp_xaa_auth import MCPXAAParams
from datarobot_genai.dragent.plugins.mcp_xaa_auth import XAAStepOneTokenExchangeParams
from datarobot_genai.dragent.plugins.mcp_xaa_auth import XAAStepTwoTokenRequestParams
from datarobot_genai.dragent.plugins.mcp_xaa_auth import (
    extract_token_value_from_bearer_or_non_bearer_header,
)
from datarobot_genai.dragent.plugins.mcp_xaa_auth import get_xaa_param_from_mcp_auth_server_metadata
from datarobot_genai.dragent.plugins.okta_a2a_auth import (
    OAuth2CrossApplicationAccessAuthProviderConfig,
)
from datarobot_genai.dragent.plugins.okta_a2a_auth import _CrossAppFlowParams


@pytest.fixture
def module_under_test() -> str:
    return "datarobot_genai.dragent.plugins.mcp_xaa_auth"


class TestXAAStepOneTokenExchangeParams:
    def test_create_from_mcp_auth_server_metadata(self) -> None:
        mock_trusted_issuer = Mock()
        mock_audience = Mock()
        mock_metadata = {"trustedIssuer": mock_trusted_issuer, "audience": mock_audience}
        output = XAAStepOneTokenExchangeParams.create_from_mcp_auth_server_metadata(mock_metadata)

        assert output.trusted_issuer == mock_trusted_issuer
        assert output.exchange_audience == mock_audience


class TestXAAStepTwoTokenRequestParams:
    def test_create_from_mcp_auth_server_metadata(self) -> None:
        mock_token_url = Mock()
        mock_audience = Mock()
        mock_scopes = Mock()
        mock_metadata = {
            "tokenUrl": mock_token_url,
            "audience": mock_audience,
            "scopes": mock_scopes,
        }
        output = XAAStepTwoTokenRequestParams.create_from_mcp_auth_server_metadata(mock_metadata)

        assert output.token_url == mock_token_url
        assert output.target_audience == mock_audience
        assert output.id_jag_scopes == mock_scopes


class TestMCPXAAParams:
    @pytest.fixture
    def mock_get_xaa_param_from_mcp_auth_server_metadata(
        self, module_under_test: str
    ) -> Iterator[AsyncMock]:
        with patch(
            f"{module_under_test}.get_xaa_param_from_mcp_auth_server_metadata",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.mark.asyncio
    async def test_get_xaa_param_from_mcp_auth_server_metadata(
        self,
        mock_get_xaa_param_from_mcp_auth_server_metadata: AsyncMock,
    ) -> None:
        mcp_auth_server_metadata_url = Mock()
        output = await MCPXAAParams.create_from_mcp_auth_server_metadata(
            mcp_auth_server_metadata_url
        )

        mock_get_xaa_param_from_mcp_auth_server_metadata.assert_called_once_with(
            mcp_auth_server_metadata_url
        )
        assert output == mock_get_xaa_param_from_mcp_auth_server_metadata.return_value


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
    def mock_parse_xaa_params_from_mcp_auth_server_metadata(
        self, module_under_test: str
    ) -> Iterator[Mock]:
        with patch(
            f"{module_under_test}.parse_xaa_params_from_mcp_auth_server_metadata",
        ) as mock_func:
            yield mock_func

    @pytest.mark.asyncio
    async def test_get_xaa_param_from_mcp_auth_server_metadata(
        self,
        mock_async_http_client: Mock,
        mock_get_retriable_async_http_client: Mock,
        mock_parse_xaa_params_from_mcp_auth_server_metadata: Mock,
    ) -> None:
        mcp_auth_server_metadata_url = Mock()
        await get_xaa_param_from_mcp_auth_server_metadata(mcp_auth_server_metadata_url)

        mock_get_retriable_async_http_client.assert_called_once_with()
        mock_async_http_client.get.assert_called_once_with(mcp_auth_server_metadata_url)
        mock_resp = mock_async_http_client.get.return_value
        mock_resp.raise_for_status.assert_called_once_with()
        mock_parse_xaa_params_from_mcp_auth_server_metadata.assert_called_once_with(
            mock_resp.json.return_value,
        )


class TestMiscellaneous:
    @pytest.mark.parametrize(
        "header_value",
        ["bearer TOKEN", "Bearer TOKEN", "TOKEN"],
        ids=str,
    )
    def test_extract_token_value_from_bearer_or_non_bearer_header(self, header_value: str) -> None:
        output = extract_token_value_from_bearer_or_non_bearer_header(header_value)
        assert output == "TOKEN"


class TestMCPXAAAuthProvider:
    @pytest.fixture
    def mock_extract_token_value_from_bearer_or_non_bearer_header(
        self,
        module_under_test: str,
    ) -> Iterator[Mock]:
        with patch(
            f"{module_under_test}.extract_token_value_from_bearer_or_non_bearer_header",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_nat_context_get(self) -> Iterator[Mock]:
        with patch.object(Context, "get") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_token_exchange(self, module_under_test: str) -> Iterator[Mock]:
        with patch(
            f"{module_under_test}.get_token_exchange",
        ) as mock_func:
            mock_func.return_value.exchange_token = AsyncMock(return_value="TOKEN")
            yield mock_func

    @pytest.fixture
    def mock_get_cross_app_flow_params(self) -> Iterator[Mock]:
        with patch.object(MCPXAAAuthProvider, "get_cross_app_flow_params") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_extract_subject_token_from_inbound_request(self) -> Iterator[Mock]:
        with patch.object(
            MCPXAAAuthProvider,
            "extract_subject_token_from_inbound_request",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_oauth2_cross_app_access_auth_provider_config(self) -> Iterator[Mock]:
        with patch.object(
            MCPXAAAuthProvider,
            "get_oauth2_cross_app_access_auth_provider_config",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_forwardable_headers_from_inbound_request(self) -> Iterator[Mock]:
        with patch.object(
            MCPXAAAuthProvider,
            "get_forwardable_headers_from_inbound_request",
        ) as mock_func:
            mock_func.return_value = [HeaderCred(name="afda", value="sdafas")]
            yield mock_func

    @pytest.fixture
    def mock_get_exchanged_token(self) -> Iterator[AsyncMock]:
        with patch.object(
            MCPXAAAuthProvider,
            "get_exchanged_token",
            new_callable=AsyncMock,
        ) as mock_func:
            mock_func.return_value = BearerTokenCred(token="adfa")
            yield mock_func

    def test_set_xaa_params(self) -> None:
        auth_provider = MCPXAAAuthProvider(Mock())
        mock_xaa_params = Mock()
        auth_provider.set_xaa_params(mock_xaa_params)

        assert auth_provider._xaa_params == mock_xaa_params

    def test_extract_subject_token_from_okta_token_header(
        self,
        mock_extract_token_value_from_bearer_or_non_bearer_header: Mock,
    ) -> None:
        auth_provider_config = Mock()
        mock_header_name = "dsfdsaas"
        auth_provider_config.okta_token_header = mock_header_name
        auth_provider_config.fallback_token_headers = [Mock()]
        mock_token_header_value = Mock()

        auth_provider = MCPXAAAuthProvider(auth_provider_config)
        output = auth_provider.extract_subject_token_from_inbound_request(
            {mock_header_name: mock_token_header_value},
        )

        mock_extract_token_value_from_bearer_or_non_bearer_header.assert_called_once_with(
            mock_token_header_value,
        )
        assert output == mock_extract_token_value_from_bearer_or_non_bearer_header.return_value

    def test_extract_subject_token_from_fallback_token_header(
        self,
        mock_extract_token_value_from_bearer_or_non_bearer_header: Mock,
    ) -> None:
        auth_provider_config = Mock()
        mock_header_name = "dsfdsaas"
        auth_provider_config.fallback_token_headers = [mock_header_name]
        mock_token_header_value = Mock()

        auth_provider = MCPXAAAuthProvider(auth_provider_config)
        output = auth_provider.extract_subject_token_from_inbound_request(
            {mock_header_name: mock_token_header_value},
        )

        mock_extract_token_value_from_bearer_or_non_bearer_header.assert_called_once_with(
            mock_token_header_value,
        )
        assert output == mock_extract_token_value_from_bearer_or_non_bearer_header.return_value

    def test_extract_subject_token_raise_error_if_no_qualified_token_header(
        self,
    ) -> None:
        auth_provider_config = Mock()
        auth_provider_config.fallback_token_headers = []

        with pytest.raises(RuntimeError):
            auth_provider = MCPXAAAuthProvider(auth_provider_config)
            auth_provider.extract_subject_token_from_inbound_request({})

    def test_get_cross_app_flow_params(self) -> None:
        auth_provider = MCPXAAAuthProvider(Mock())
        xaa_params = Mock()
        auth_provider.set_xaa_params(xaa_params)

        output = auth_provider.get_cross_app_flow_params()
        assert output == _CrossAppFlowParams(
            xaa_params.step_two_token_request_params.token_url,
            xaa_params.step_one_token_exchange_params.trusted_issuer,
            xaa_params.step_one_token_exchange_params.exchange_audience,
            xaa_params.step_two_token_request_params.target_audience,
            "private_key_jwt",
            xaa_params.step_two_token_request_params.id_jag_scopes,
        )

    def test_get_oauth2_cross_app_access_auth_provider_config(self) -> None:
        auth_provider_config = MCPXAAAuthProviderConfig(
            okta_token_header="adfaadf",
            fallback_token_headers=["dafdaas"],
            principal_id="dsafa",
            private_jwk="dasdfae",
        )
        auth_provider = MCPXAAAuthProvider(auth_provider_config)
        output = auth_provider.get_oauth2_cross_app_access_auth_provider_config()

        assert output == OAuth2CrossApplicationAccessAuthProviderConfig(
            okta_token_header=auth_provider_config.okta_token_header,
            fallback_token_headers=auth_provider_config.fallback_token_headers,
            principal_id=auth_provider_config.principal_id,
            private_jwk=auth_provider_config.private_jwk,
        )

    def test_get_non_forwardable_header_keys(self) -> None:
        auth_provider_config = MCPXAAAuthProviderConfig()
        auth_provider = MCPXAAAuthProvider(auth_provider_config)

        output = auth_provider.get_non_forwardable_header_keys()
        assert output == {"x-datarobot-external-access-token", "authorization"}

    def test_get_forwardable_headers_from_inbound_request(self) -> None:
        auth_provider_config = MCPXAAAuthProviderConfig()
        auth_provider = MCPXAAAuthProvider(auth_provider_config)
        headers = {"afda": "sdafas"}
        output = auth_provider.get_forwardable_headers_from_inbound_request(headers)

        assert output == [HeaderCred(name="afda", value=SecretStr(headers["afda"]))]

    @pytest.mark.asyncio
    async def test_authenticate(
        self,
        mock_get_forwardable_headers_from_inbound_request: Mock,
        mock_get_exchanged_token: AsyncMock,
        mock_nat_context_get: Mock,
    ) -> None:
        inbound_headers = Mock()
        mock_nat_context_get.return_value.metadata.headers = inbound_headers
        auth_provider = MCPXAAAuthProvider(MCPXAAAuthProviderConfig())
        auth_provider.set_xaa_params(Mock())
        output = await auth_provider.authenticate()

        mock_get_forwardable_headers_from_inbound_request.assert_called_once_with(inbound_headers)
        mock_get_exchanged_token.assert_called_once_with(inbound_headers)
        assert output == AuthResult(
            credentials=[
                *mock_get_forwardable_headers_from_inbound_request.return_value,
                mock_get_exchanged_token.return_value,
            ]
        )

    @pytest.mark.asyncio
    async def test_authenticate_raises_error_when_xaa_params_not_set(
        self,
        mock_get_cross_app_flow_params: Mock,
    ) -> None:
        auth_provider = MCPXAAAuthProvider(Mock())

        with pytest.raises(RuntimeError):
            assert auth_provider._xaa_params is None
            await auth_provider.authenticate()
