from collections.abc import Iterator
from http import HTTPMethod
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import datarobot as dr
import pytest

from datarobot_genai.drmcpbase.datarobot_services.client import DataRobotClientWithAsyncAPI
from datarobot_genai.drmcpbase.datarobot_services.client import get_async_https_retry_client
from datarobot_genai.drmcpbase.datarobot_services.client import get_async_https_session
from datarobot_genai.drmcpbase.datarobot_services.client import get_ssl_context_from_ca_file


@pytest.fixture
def module_under_test() -> str:
    return "datarobot_genai.drmcpbase.datarobot_services.client"


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

    @pytest.mark.asyncio
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


class TestConnectionSetupRelated:
    @pytest.fixture
    def mock_create_default_ssl_context(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.create_default_ssl_context") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_tcp_connector_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.TCPConnector") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_client_timeout_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.ClientTimeout") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_client_session_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.ClientSession") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_exponential_retry_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.ExponentialRetry") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_retry_client_cls(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.RetryClient") as mock_cls:
            yield mock_cls

    @pytest.fixture
    def mock_get_ssl_context_from_ca_file(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_ssl_context_from_ca_file") as mock_func:
            yield mock_func

    def test_ssl_context_from_ca_file(self, mock_create_default_ssl_context: Mock) -> None:
        mock_ca_path = Mock()
        output = get_ssl_context_from_ca_file(mock_ca_path)

        mock_create_default_ssl_context.assert_called_once_with()
        mock_ssl_context = mock_create_default_ssl_context.return_value
        mock_ssl_context.load_verify_locations.assert_called_once_with(cafile=mock_ca_path)
        assert output == mock_create_default_ssl_context.return_value

    def test_get_async_https_session(
        self,
        mock_tcp_connector_cls: Mock,
        mock_client_timeout_cls: Mock,
        mock_client_session_cls: Mock,
        mock_get_ssl_context_from_ca_file: Mock,
    ) -> None:
        mock_ca = Mock()
        output = get_async_https_session(mock_ca)

        mock_get_ssl_context_from_ca_file.assert_called_once_with(mock_ca)
        mock_ssl_context = mock_get_ssl_context_from_ca_file.return_value
        mock_tcp_connector_cls.assert_called_once_with(ssl=mock_ssl_context)
        mock_client_timeout_cls.assert_called_once_with(connect=30, sock_read=60)
        expected_headers = {"User-Agent": "global-mcp"}
        mock_client_session_cls.assert_called_once_with(
            headers=expected_headers,
            connector=mock_tcp_connector_cls.return_value,
            timeout=mock_client_timeout_cls.return_value,
        )
        assert output == mock_client_session_cls.return_value

    def test_get_async_https_retry_client(
        self,
        mock_exponential_retry_cls: Mock,
        mock_retry_client_cls: Mock,
    ) -> None:
        mock_session = Mock()
        output = get_async_https_retry_client(mock_session)

        mock_exponential_retry_cls.assert_called_once_with(
            attempts=3,
            start_timeout=0.1,
            methods=[HTTPMethod.GET, HTTPMethod.POST, HTTPMethod.PUT],
        )
        mock_retry = mock_exponential_retry_cls.return_value
        mock_retry_client_cls.assert_called_once_with(
            client_session=mock_session,
            retry_options=mock_retry,
        )
        assert output == mock_retry_client_cls.return_value
