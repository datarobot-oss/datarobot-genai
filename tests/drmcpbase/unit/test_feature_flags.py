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
from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcpbase.auth.enums import DataRobotBearerHeaderEnum
from datarobot_genai.drmcpbase.auth.exceptions import InvalidBearerTokenError
from datarobot_genai.drmcpbase.auth.exceptions import (
    NoDataRobotBearerTokenFoundInRequestContextError,
)
from datarobot_genai.drmcpbase.auth.exceptions import NoHeadersFoundInRequestContextError
from datarobot_genai.drmcpbase.feature_flags import FeatureFlagEvaluation
from datarobot_genai.drmcpbase.feature_flags import check_mcp_tools_gallery_support


@pytest.fixture
def module_under_test():
    return "datarobot_genai.drmcpbase.feature_flags"


class TestFeatureFlagEvaluation:
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
    def mock_get_feature_flag_enablement_with_existing_datarobot_client(
        self, module_under_test: str
    ) -> Iterator[AsyncMock]:
        with patch(
            f"{module_under_test}.get_feature_flag_enablement_with_existing_datarobot_client",
        ) as mock_func:
            yield mock_func

    @pytest.mark.asyncio
    async def test_is_mcp_tools_gallery_support_enabled_for_user_in_mcp_request(
        self,
        mock_get_datarobot_bearer_token_from_x_datarobot_authorization: Mock,
        mock_get_feature_flag_enablement_with_existing_datarobot_client: AsyncMock,
    ) -> None:
        mock_datarobot_api_client = Mock()
        output = await FeatureFlagEvaluation.is_mcp_tools_gallery_support_enabled_for_user_in_mcp_request(  # noqa: E501
            mock_datarobot_api_client
        )

        mock_get_datarobot_bearer_token_from_x_datarobot_authorization.assert_called_once_with()
        mock_datarobot_api_token = (
            mock_get_datarobot_bearer_token_from_x_datarobot_authorization.return_value
        )
        mock_get_feature_flag_enablement_with_existing_datarobot_client.assert_called_once_with(
            mock_datarobot_api_client,
            mock_datarobot_api_token,
            "ENABLE_MCP_TOOLS_GALLERY_SUPPORT",
        )
        assert (
            output == mock_get_feature_flag_enablement_with_existing_datarobot_client.return_value
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raised_error",
        [
            NoHeadersFoundInRequestContextError,
            NoDataRobotBearerTokenFoundInRequestContextError,
            InvalidBearerTokenError,
        ],
        ids=str,
    )
    async def test_is_mcp_tools_gallery_support_enabled_for_user_in_mcp_request_fall_back_to_false_if_errored(  # noqa: E501
        self,
        raised_error: NoHeadersFoundInRequestContextError
        | NoDataRobotBearerTokenFoundInRequestContextError,
        mock_get_datarobot_bearer_token_from_x_datarobot_authorization: Mock,
        mock_get_feature_flag_enablement_with_existing_datarobot_client: AsyncMock,
    ) -> None:
        mock_get_datarobot_bearer_token_from_x_datarobot_authorization.side_effect = raised_error

        mock_datarobot_api_client = Mock()
        output = await FeatureFlagEvaluation.is_mcp_tools_gallery_support_enabled_for_user_in_mcp_request(  # noqa: E501
            mock_datarobot_api_client
        )

        mock_get_datarobot_bearer_token_from_x_datarobot_authorization.assert_called_once_with()
        mock_get_feature_flag_enablement_with_existing_datarobot_client.assert_not_called()
        assert output is False


class TestCheckMcpToolsGallerySupport:
    """The shared None-safe entry point used by both providers and the gallery gate."""

    @pytest.fixture
    def mock_evaluator(self, module_under_test: str) -> Iterator[AsyncMock]:
        with patch(
            f"{module_under_test}.FeatureFlagEvaluation."
            "is_mcp_tools_gallery_support_enabled_for_user_in_mcp_request",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.mark.asyncio
    async def test_none_client_returns_false_without_evaluating(
        self, mock_evaluator: AsyncMock
    ) -> None:
        # A request racing a provider's lifespan has no client yet → fail closed,
        # and the evaluator (which reads request headers) must not be called.
        assert await check_mcp_tools_gallery_support(None) is False
        mock_evaluator.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("enabled", [True, False])
    async def test_delegates_to_evaluator_with_client(
        self, mock_evaluator: AsyncMock, enabled: bool
    ) -> None:
        mock_evaluator.return_value = enabled
        client = Mock()

        assert await check_mcp_tools_gallery_support(client) is enabled
        mock_evaluator.assert_awaited_once_with(client)
