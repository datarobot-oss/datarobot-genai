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

from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import datarobot as dr
import pytest

from datarobot_genai.drtools.core.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.core.clients.datarobot import DataRobotClientWithAsyncAPI
from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.exceptions import ToolError


@pytest.fixture
def module_under_test() -> str:
    return "datarobot_genai.drtools.core.clients.datarobot"


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
    @patch("datarobot_genai.drtools.core.clients.datarobot.get_credentials")
    def test_get_client_calls_dr_client_with_token_and_endpoint(
        self, mock_get_credentials: MagicMock, mock_dr: MagicMock, mock_dr_context: MagicMock
    ) -> None:
        """Test that get_client configures dr.Client with token and endpoint from credentials."""
        mock_creds = MagicMock()
        mock_creds.datarobot.endpoint = "https://app.datarobot.com/api/v2"
        mock_get_credentials.return_value = mock_creds

        client = DataRobotClient("token-from-headers")
        result = client.get_client()

        mock_dr.Client.assert_called_once_with(
            token="token-from-headers",
            endpoint="https://app.datarobot.com/api/v2",
        )
        assert result is mock_dr

    @patch("datarobot_genai.drtools.core.clients.datarobot.DRContext")
    @patch("datarobot_genai.drtools.core.clients.datarobot.dr")
    @patch("datarobot_genai.drtools.core.clients.datarobot.get_credentials")
    def test_get_client_resets_dr_context_use_case(
        self, mock_get_credentials: MagicMock, mock_dr: MagicMock, mock_dr_context: MagicMock
    ) -> None:
        """Test that get_client sets DRContext.use_case to None."""
        mock_creds = MagicMock()
        mock_creds.datarobot.endpoint = "https://app.datarobot.com/api/v2"
        mock_get_credentials.return_value = mock_creds

        client = DataRobotClient("token")
        client.get_client()

        # Implementation does DRContext.use_case = None
        assert mock_dr_context.use_case is None

    @patch("datarobot_genai.drtools.core.clients.datarobot.DRContext")
    @patch("datarobot_genai.drtools.core.clients.datarobot.dr")
    @patch("datarobot_genai.drtools.core.clients.datarobot.get_credentials")
    def test_get_client_returns_dr_module(
        self, mock_get_credentials: MagicMock, mock_dr: MagicMock, mock_dr_context: MagicMock
    ) -> None:
        """Test that get_client returns the datarobot module."""
        mock_creds = MagicMock()
        mock_creds.datarobot.endpoint = "https://example.com/api/v2"
        mock_get_credentials.return_value = mock_creds

        client = DataRobotClient("any-token")
        result = client.get_client()

        assert result is mock_dr


class TestDataRobotClientWithAsyncAPI:
    @pytest.fixture
    def mock_get_api_v2_endpoint(self) -> Iterator[Mock]:
        with patch.object(DataRobotClientWithAsyncAPI, "get_api_v2_endpoint") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_https_response(self) -> Mock:
        mock_resp = Mock()
        mock_resp.text = AsyncMock(return_value="")
        mock_resp.json = AsyncMock(return_value={})
        return mock_resp

    @pytest.fixture
    def mock_https_action_async_context(self, mock_https_response: Mock) -> AsyncMock:
        https_action_context = AsyncMock()
        https_action_context.__aenter__.return_value = mock_https_response
        https_action_context.__aexit__.return_value = None
        return https_action_context

    @pytest.fixture
    def mock_https_client_with_action_returning_async_context(
        self, mock_https_action_async_context: AsyncMock
    ) -> Mock:
        mock_client = Mock()
        mock_client.get = Mock(return_value=mock_https_action_async_context)
        mock_client.post = Mock(return_value=mock_https_action_async_context)
        mock_client.close = AsyncMock()
        return mock_client

    @pytest.fixture
    def mock_get_async_https_retry_client(
        self,
        module_under_test: str,
        mock_https_client_with_action_returning_async_context: Mock,
    ) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_async_https_retry_client") as mock_func:
            mock_func.return_value = mock_https_client_with_action_returning_async_context
            yield mock_func

    @pytest.fixture
    def mock_get_async_https_session(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_async_https_session") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_feature_entitlement_evaluate_result(self) -> Iterator[Mock]:
        with patch.object(
            DataRobotClientWithAsyncAPI,
            "get_feature_entitlement_evaluate_result",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_unpaginate(self) -> Iterator[Mock]:
        with patch.object(DataRobotClientWithAsyncAPI, "_unpaginate") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_clean_up(self) -> Iterator[Mock]:
        with patch.object(
            DataRobotClientWithAsyncAPI,
            "clean_up",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.mark.parametrize("host_url", ["https://dafaasfa", "https://dafaasfa/api/v2/"], ids=str)
    def test_get_api_v2_endpoint_with_various_host_urls(self, host_url: str) -> None:
        output = DataRobotClientWithAsyncAPI.get_api_v2_endpoint(host_url, "AAA")
        assert output == "https://dafaasfa/api/v2/AAA/"

    @pytest.mark.parametrize("v2_api_path", ["AAA", "/AAA"], ids=str)
    def test_get_api_v2_endpoint_with_various_path_urls(self, v2_api_path: str) -> None:
        host_url = "https://dafaasfa"
        output = DataRobotClientWithAsyncAPI.get_api_v2_endpoint(host_url, v2_api_path)
        assert output == f"{host_url}/api/v2/AAA/"

    def test_init(
        self,
        mock_get_async_https_retry_client: Mock,
        mock_get_async_https_session: Mock,
    ) -> None:
        mock_host = Mock()
        api_client = DataRobotClientWithAsyncAPI(mock_host)

        assert api_client._dr_host == mock_host
        mock_get_async_https_session.assert_called_once_with()
        mock_session = mock_get_async_https_session.return_value
        assert api_client._session == mock_session
        mock_get_async_https_retry_client.assert_called_once_with(mock_session)
        assert api_client._retry_client == mock_get_async_https_retry_client.return_value

    @pytest.mark.usefixtures(
        "mock_get_async_https_retry_client",
        "mock_get_async_https_session",
    )
    async def test_clean_up(
        self,
        mock_https_client_with_action_returning_async_context: Mock,
    ) -> None:
        api_client = DataRobotClientWithAsyncAPI(Mock())

        await api_client.clean_up()

        mock_https_client_with_action_returning_async_context.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_client_as_context_manager(self, mock_clean_up: AsyncMock) -> None:
        async with DataRobotClientWithAsyncAPI(Mock()) as api_client:
            pass

        assert isinstance(api_client, DataRobotClientWithAsyncAPI)
        mock_clean_up.assert_called_once_with()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_get_async_https_retry_client",
        "mock_get_async_https_session",
    )
    async def test_get_feature_entitlement_evaluate_result(
        self,
        mock_get_api_v2_endpoint: Mock,
        mock_https_client_with_action_returning_async_context: Mock,
        mock_https_response: Mock,
    ) -> None:
        mock_dr_host = Mock()
        api_client = DataRobotClientWithAsyncAPI(mock_dr_host)
        feature_flag_name = Mock()
        mock_token = Mock()
        output = await api_client.get_feature_entitlement_evaluate_result(
            feature_flag_name, mock_token
        )

        mock_get_api_v2_endpoint.assert_called_once_with(mock_dr_host, "/entitlements/evaluate/")
        expected_url = mock_get_api_v2_endpoint.return_value
        mock_https_client_with_action_returning_async_context.post.assert_called_once_with(
            expected_url,
            json={"entitlements": [{"name": feature_flag_name}]},
            headers={"Authorization": f"Bearer {mock_token}"},
        )
        assert output == mock_https_response.json.return_value

    @pytest.mark.asyncio
    @pytest.mark.parametrize("feature_flag_value", [True, False], ids=str)
    async def test_is_feature_flag_enabled(
        self,
        feature_flag_value: bool,
        mock_get_feature_entitlement_evaluate_result: Mock,
    ) -> None:
        mock_get_feature_entitlement_evaluate_result.return_value = {
            "entitlements": [{"value": feature_flag_value}]
        }

        api_client = DataRobotClientWithAsyncAPI(Mock())
        feature_flag_name = Mock()
        mock_token = Mock()
        output = await api_client.is_feature_flag_enabled(feature_flag_name, mock_token)

        mock_get_feature_entitlement_evaluate_result.assert_called_once_with(
            feature_flag_name,
            mock_token,
        )
        assert output is feature_flag_value

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_get_async_https_retry_client",
        "mock_get_async_https_session",
    )
    async def test_unpaginate(
        self,
        mock_https_client_with_action_returning_async_context: Mock,
    ) -> None:
        api_client = DataRobotClientWithAsyncAPI(Mock())
        mock_url = Mock()
        mock_headers = Mock()
        async for _ in api_client._unpaginate(mock_url, mock_headers):
            pass

        mock_https_client_with_action_returning_async_context.get.assert_called_once_with(
            mock_url, headers=mock_headers
        )

    @pytest.mark.asyncio
    async def test_list_mcp_deployment_ids_only_return_mcp_deployment(
        self,
        mock_get_api_v2_endpoint: Mock,
        mock_unpaginate: Mock,
    ) -> None:
        mcp_deployment = {"id": Mock(), "model": {"targetType": "MCP"}}
        non_mcp_deployment = {"id": Mock(), "model": {"targetType": Mock()}}

        async def fake_unpaginate(*args: object, **kwargs: object) -> object:
            yield mcp_deployment
            yield non_mcp_deployment

        mock_unpaginate.side_effect = fake_unpaginate

        mock_dr_host = Mock()
        api_client = DataRobotClientWithAsyncAPI(mock_dr_host)
        mock_token = Mock()
        outputs = await api_client.list_mcp_deployment_ids(mock_token)

        mock_get_api_v2_endpoint.assert_called_once_with(mock_dr_host, "/deployments/")
        expected_url = mock_get_api_v2_endpoint.return_value
        expected_headers = {"Authorization": f"Bearer {mock_token}"}
        mock_unpaginate.assert_called_once_with(expected_url, expected_headers)
        assert outputs == [mcp_deployment["id"]]

    @pytest.mark.asyncio
    async def test_list_mcp_tool_custom_model_deployment_ids(
        self,
        mock_get_api_v2_endpoint: Mock,
        mock_unpaginate: Mock,
    ) -> None:
        api_v2_endpoint = "https://foo/bar"
        mock_get_api_v2_endpoint.return_value = api_v2_endpoint
        deployment_with_tool_as_tag_name = {"id": Mock(), "tags": [{"name": "tool"}]}
        deployment_with_tool_as_tag_value = {"id": Mock(), "tags": [{"value": "tool"}]}
        deployment_with_tool_as_both_tag_key_and_value = {
            "id": Mock(),
            "tags": [{"name": "tool", "value": "tool"}],
        }
        non_tool_deployment = {"id": Mock()}

        async def fake_unpaginate(*args: object, **kwargs: object) -> object:
            yield deployment_with_tool_as_tag_name
            yield deployment_with_tool_as_tag_value
            yield deployment_with_tool_as_both_tag_key_and_value
            yield non_tool_deployment

        mock_unpaginate.side_effect = fake_unpaginate

        mock_dr_host = Mock()
        api_client = DataRobotClientWithAsyncAPI(mock_dr_host)
        mock_token = Mock()
        outputs = await api_client.list_mcp_tool_custom_model_deployment_ids(mock_token)

        mock_get_api_v2_endpoint.assert_called_once_with(mock_dr_host, "/deployments")
        expected_url = f"{api_v2_endpoint}?tagValues=tool&tagKeys=tool"
        expected_headers = {"Authorization": f"Bearer {mock_token}"}
        mock_unpaginate.assert_called_once_with(expected_url, expected_headers)
        assert outputs == [deployment_with_tool_as_both_tag_key_and_value["id"]]

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_get_async_https_retry_client",
        "mock_get_async_https_session",
    )
    async def test_get_datarobot_deployment(
        self,
        mock_get_api_v2_endpoint: Mock,
        mock_https_client_with_action_returning_async_context: Mock,
        mock_https_response: Mock,
    ) -> None:
        expected_deployment_id = "dsafa"
        mock_https_response.json = AsyncMock(return_value={"id": expected_deployment_id})

        mock_dr_host = Mock()
        api_client = DataRobotClientWithAsyncAPI(mock_dr_host)
        mock_deployment_id = Mock()
        mock_token = Mock()
        output = await api_client.get_datarobot_deployment(mock_deployment_id, mock_token)

        mock_get_api_v2_endpoint.assert_called_once_with(
            mock_dr_host,
            f"/deployments/{mock_deployment_id}/",
        )
        expected_headers = {"Authorization": f"Bearer {mock_token}"}
        mock_https_client_with_action_returning_async_context.get.assert_called_once_with(
            mock_get_api_v2_endpoint.return_value,
            headers=expected_headers,
        )
        assert isinstance(output, dr.Deployment)
        assert output.id == expected_deployment_id
