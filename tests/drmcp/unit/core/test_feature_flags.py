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

from datarobot_genai.drmcp.core.feature_flags import FeatureFlag


class TestFeatureFlags:
    @pytest.fixture
    def module_under_test(self) -> str:
        return "datarobot_genai.drmcp.core.feature_flags"

    @pytest.fixture
    def mock_get_credentials(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_credentials") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_datarobot_client_with_async_api_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.DataRobotClientWithAsyncAPI") as mock_cls:
            yield mock_cls

    @pytest.mark.asyncio
    async def test_is_mcp_tools_gallery_support_enabled_for_static_mcp_container_user(
        self,
        mock_get_credentials: Mock,
        mock_datarobot_client_with_async_api_cls: Mock,
    ) -> None:
        mock_api_client = (
            mock_datarobot_client_with_async_api_cls.return_value.__aenter__.return_value
        )
        expected_feature_flag_value = Mock()
        mock_api_client.is_feature_flag_enabled = AsyncMock(
            return_value=expected_feature_flag_value
        )

        output = (
            await FeatureFlag.is_mcp_tools_gallery_support_enabled_for_static_mcp_container_user()
        )

        mock_credentials = mock_get_credentials.return_value
        mock_token = mock_credentials.datarobot.application_api_token
        mock_endpoint = mock_credentials.datarobot.endpoint
        mock_datarobot_client_with_async_api_cls.assert_called_once_with(mock_endpoint)
        mock_api_client.is_feature_flag_enabled.assert_called_once_with(
            "ENABLE_MCP_TOOLS_GALLERY_SUPPORT",
            mock_token,
        )
        assert output == expected_feature_flag_value
