from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcpbase.datarobot_services.feature_flags import (
    is_mcp_tools_gallery_support_enabled,
)


class TestFeatureFlags:
    @pytest.fixture
    def module_under_test(self) -> str:
        return "datarobot_genai.drmcpbase.datarobot_services.feature_flags"

    @pytest.fixture
    def mock_datarobot_client_with_async_api_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.DataRobotClientWithAsyncAPI") as mock_cls:
            yield mock_cls

    @pytest.mark.asyncio
    async def test_is_mcp_tools_gallery_support_enabled(
        self,
        mock_datarobot_client_with_async_api_cls: Mock,
    ) -> None:
        mock_api_client = (
            mock_datarobot_client_with_async_api_cls.return_value.__aenter__.return_value
        )
        expected_feature_flag_value = Mock()
        mock_api_client.is_feature_flag_enabled = AsyncMock(
            return_value=expected_feature_flag_value
        )

        mock_api_host = Mock()
        mock_api_token = Mock()
        output = await is_mcp_tools_gallery_support_enabled(mock_api_host, mock_api_token)

        mock_datarobot_client_with_async_api_cls.assert_called_once_with(mock_api_host)
        mock_api_client.is_feature_flag_enabled.assert_called_once_with(
            "ENABLE_MCP_TOOLS_GALLERY_SUPPORT",
            mock_api_token,
        )
        assert output == expected_feature_flag_value
