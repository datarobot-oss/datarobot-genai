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

from datarobot_genai.dragent.plugins.user_mcp_xxa_client import (
    get_xaa_param_from_mcp_auth_server_metadata,
)
from datarobot_genai.dragent.plugins.user_mcp_xxa_client import (
    parse_xaa_params_from_mcp_auth_server_metadata,
)
from datarobot_genai.dragent.plugins.xaa_auth import XAAParams
from datarobot_genai.dragent.plugins.xaa_auth import XAAStepOneTokenExchangeParams
from datarobot_genai.dragent.plugins.xaa_auth import XAAStepTwoTokenRequestParams


@pytest.fixture
def module_under_test() -> str:
    return "datarobot_genai.dragent.plugins.user_mcp_xxa_client"


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

    def test_parse_xaa_params_from_mcp_auth_server_metadata(self) -> None:
        mock_trust_issuer = Mock()
        mock_exchange_audience = Mock()
        mock_token_url = Mock()
        mock_target_audience = Mock()
        mock_scopes = [Mock()]
        mcp_auth_server_metadata = {
            "urn:datarobot:nat_mcp_xaa_client": {
                "tokenExchange": {
                    "trustedIssuer": mock_trust_issuer,
                    "audience": mock_exchange_audience,
                },
                "tokenRequest": {
                    "tokenUrl": mock_token_url,
                    "audience": mock_target_audience,
                    "scopes": mock_scopes,
                },
            }
        }
        output = parse_xaa_params_from_mcp_auth_server_metadata(mcp_auth_server_metadata)
        assert output == XAAParams(
            XAAStepOneTokenExchangeParams(mock_trust_issuer, mock_exchange_audience),
            XAAStepTwoTokenRequestParams(
                mock_token_url,
                mock_target_audience,
                mock_scopes,
            ),
        )

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
