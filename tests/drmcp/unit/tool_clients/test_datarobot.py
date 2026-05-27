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

"""Tests for DataRobot tools client (get_datarobot_access_token, DataRobotClient)."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.drtools.core.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.exceptions import ToolError


class TestGetDatarobotAccessToken:
    """Test get_datarobot_access_token function."""

    @pytest.mark.asyncio
    async def test_returns_token_when_resolve_token_from_headers_returns_token(self) -> None:
        """Test successful token retrieval from headers."""
        with patch(
            "datarobot_genai.drtools.core.clients.datarobot.resolve_token_from_headers",
            return_value="bearer-token-123",
        ):
            result = await get_datarobot_access_token()
        assert result == "bearer-token-123"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_raises_tool_error_when_no_token(self) -> None:
        """Test that ToolError is raised when resolve_token_from_headers returns None."""
        with patch(
            "datarobot_genai.drtools.core.clients.datarobot.resolve_token_from_headers",
            return_value=None,
        ):
            with pytest.raises(ToolError) as exc_info:
                await get_datarobot_access_token()
        assert "DataRobot API token not found" in str(exc_info.value)
        assert "Authorization" in str(exc_info.value) or "x-datarobot-api-token" in str(
            exc_info.value
        )

    @pytest.mark.asyncio
    async def test_raises_tool_error_when_empty_token(self) -> None:
        """Test that ToolError is raised when resolve_token_from_headers returns empty string."""
        with patch(
            "datarobot_genai.drtools.core.clients.datarobot.resolve_token_from_headers",
            return_value="",
        ):
            with pytest.raises(ToolError) as exc_info:
                await get_datarobot_access_token()
        assert "DataRobot API token not found" in str(exc_info.value)


class TestDataRobotClient:
    """Test DataRobotClient class."""

    def test_init_stores_token(self) -> None:
        """Test that DataRobotClient stores the token."""
        client = DataRobotClient("my-token")
        assert client._token == "my-token"

    @patch("datarobot_genai.drtools.core.clients.datarobot.DRContext")
    @patch("datarobot_genai.drtools.core.clients.datarobot.dr")
    @patch("datarobot_genai.drtools.core.clients.datarobot.client_configuration")
    @patch("datarobot_genai.drtools.core.clients.datarobot.get_credentials")
    def test_get_client_calls_client_configuration_with_token_and_endpoint(
        self,
        mock_get_credentials: MagicMock,
        mock_client_config: MagicMock,
        mock_dr: MagicMock,
        mock_dr_context: MagicMock,
    ) -> None:
        """Test that get_client uses client_configuration with the correct token and endpoint."""
        mock_creds = MagicMock()
        mock_creds.datarobot.endpoint = "https://app.datarobot.com/api/v2"
        mock_get_credentials.return_value = mock_creds
        mock_client_config.return_value.__enter__ = MagicMock(return_value=None)
        mock_client_config.return_value.__exit__ = MagicMock(return_value=False)

        client = DataRobotClient("token-from-headers")
        with client.get_client():
            pass

        mock_client_config.assert_called_once_with(
            token="token-from-headers",
            endpoint="https://app.datarobot.com/api/v2",
        )

    @patch("datarobot_genai.drtools.core.clients.datarobot.DRContext")
    @patch("datarobot_genai.drtools.core.clients.datarobot.dr")
    @patch("datarobot_genai.drtools.core.clients.datarobot.client_configuration")
    @patch("datarobot_genai.drtools.core.clients.datarobot.get_credentials")
    def test_get_client_resets_dr_context_use_case(
        self,
        mock_get_credentials: MagicMock,
        mock_client_config: MagicMock,
        mock_dr: MagicMock,
        mock_dr_context: MagicMock,
    ) -> None:
        """Test that get_client sets DRContext.use_case to None inside the context."""
        mock_creds = MagicMock()
        mock_creds.datarobot.endpoint = "https://app.datarobot.com/api/v2"
        mock_get_credentials.return_value = mock_creds
        mock_client_config.return_value.__enter__ = MagicMock(return_value=None)
        mock_client_config.return_value.__exit__ = MagicMock(return_value=False)

        client = DataRobotClient("token")
        with client.get_client():
            assert mock_dr_context.use_case is None

    @patch("datarobot_genai.drtools.core.clients.datarobot.DRContext")
    @patch("datarobot_genai.drtools.core.clients.datarobot.dr")
    @patch("datarobot_genai.drtools.core.clients.datarobot.client_configuration")
    @patch("datarobot_genai.drtools.core.clients.datarobot.get_credentials")
    def test_get_client_yields_dr_module(
        self,
        mock_get_credentials: MagicMock,
        mock_client_config: MagicMock,
        mock_dr: MagicMock,
        mock_dr_context: MagicMock,
    ) -> None:
        """Test that get_client yields the datarobot module."""
        mock_creds = MagicMock()
        mock_creds.datarobot.endpoint = "https://example.com/api/v2"
        mock_get_credentials.return_value = mock_creds
        mock_client_config.return_value.__enter__ = MagicMock(return_value=None)
        mock_client_config.return_value.__exit__ = MagicMock(return_value=False)

        client = DataRobotClient("any-token")
        with client.get_client() as result:
            assert result is mock_dr
