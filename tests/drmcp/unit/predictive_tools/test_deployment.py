# Copyright 2025 DataRobot, Inc.
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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from fastmcp.exceptions import ToolError

from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.tools.predictive import deployment


def test_load_dotenv() -> None:
    load_dotenv(verbose=True)


@pytest.mark.asyncio
async def test_list_deployments_success() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.deployment.get_sdk_client"
    ) as mock_get_client:
        mock_client = MagicMock()
        mock_dep1 = MagicMock(id="1", label="dep1")
        mock_dep2 = MagicMock(id="2", label="dep2")
        mock_client.Deployment.list.return_value = [mock_dep1, mock_dep2]
        mock_get_client.return_value = mock_client
        result = await deployment.list_deployments()
        assert result.structured_content["deployments"]["1"] == "dep1"
        assert result.structured_content["deployments"]["2"] == "dep2"


@pytest.mark.asyncio
async def test_list_deployments_empty() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.deployment.get_sdk_client"
    ) as mock_get_client:
        mock_client = MagicMock()
        mock_client.Deployment.list.return_value = []
        mock_get_client.return_value = mock_client
        result = await deployment.list_deployments()
        assert result.structured_content["deployments"] == []


@pytest.mark.asyncio
async def test_list_deployments_error() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.deployment.get_sdk_client",
        side_effect=Exception("fail"),
    ):
        with pytest.raises(ToolError) as exc_info:
            await deployment.list_deployments()
        assert "Error in list_deployments: Exception: fail" == str(exc_info.value)


@pytest.mark.asyncio
async def test_get_model_info_from_deployment_success() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.deployment.get_sdk_client"
    ) as mock_get_client:
        mock_client = MagicMock()
        mock_deployment = MagicMock()
        mock_deployment.model = {"project_id": "pid", "model_id": "mid"}
        mock_client.Deployment.get.return_value = mock_deployment
        mock_get_client.return_value = mock_client
        result = await deployment.get_model_info_from_deployment(deployment_id="dep_id")
        mock_client.Deployment.get.assert_called_once_with("dep_id")
        assert "project_id" in result.content[0].text
        assert result.structured_content["project_id"] == "pid"


@pytest.mark.asyncio
async def test_get_model_info_from_deployment_not_found() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.deployment.get_sdk_client"
    ) as mock_get_client:
        mock_client = MagicMock()
        mock_client.Deployment.get.side_effect = Exception(
            "404 client error: {'message': 'Not Found'}"
        )
        mock_get_client.return_value = mock_client
        with pytest.raises(ToolError) as exc_info:
            await deployment.get_model_info_from_deployment(deployment_id="dep_id")
        assert (
            "Error in get_model_info_from_deployment: Exception: 404 client error: "
            "{'message': 'Not Found'}" == str(exc_info.value)
        )


@pytest.mark.asyncio
async def test_get_model_info_from_deployment_error() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.deployment.get_sdk_client",
        side_effect=Exception("fail"),
    ):
        with pytest.raises(ToolError) as exc_info:
            await deployment.get_model_info_from_deployment(deployment_id="dep_id")
        assert "Error in get_model_info_from_deployment: Exception: fail" == str(exc_info.value)


@pytest.mark.asyncio
async def test_deploy_model_success() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.deployment.get_sdk_client"
    ) as mock_get_client:
        mock_client = MagicMock()
        mock_server = MagicMock(id="srv1")
        mock_client.PredictionServer.list.return_value = [mock_server]
        mock_deployment = MagicMock(id="dep123")
        mock_client.Deployment.create_from_learning_model.return_value = mock_deployment
        mock_get_client.return_value = mock_client
        result = await deployment.deploy_model(
            model_id="model123", label="Test Deployment", description="desc"
        )
        assert result.structured_content["deployment_id"] == "dep123"
        assert result.structured_content["label"] == "Test Deployment"


@pytest.mark.asyncio
async def test_deploy_model_no_prediction_servers() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.deployment.get_sdk_client"
    ) as mock_get_client:
        mock_client = MagicMock()
        mock_client.PredictionServer.list.return_value = []
        mock_get_client.return_value = mock_client
        with pytest.raises(ToolError) as exc_info:
            await deployment.deploy_model(model_id="model123", label="Test Deployment")
        assert "No prediction servers available" in str(exc_info.value)


@pytest.mark.asyncio
async def test_deploy_model_error() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.deployment.get_sdk_client"
    ) as mock_get_client:
        mock_client = MagicMock()
        mock_client.PredictionServer.list.side_effect = Exception("fail servers")
        mock_get_client.return_value = mock_client
        with pytest.raises(ToolError) as exc_info:
            await deployment.deploy_model(model_id="model123", label="Test Deployment")
        assert "fail servers" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_sdk_client_uses_bearer_token() -> None:
    with (
        patch("datarobot.Client") as mock_client,
        patch("datarobot_genai.drmcp.core.clients.get_http_headers") as mock_get_headers,
        patch("datarobot_genai.drmcp.core.clients.get_credentials") as mock_get_creds,
    ):
        mock_get_headers.return_value = {"authorization": "Bearer test-user-token"}
        mock_creds = MagicMock()
        mock_creds.datarobot.endpoint = "test-endpoint"
        mock_get_creds.return_value = mock_creds
        get_sdk_client()
        mock_client.assert_called_once()
        args, kwargs = mock_client.call_args
        assert kwargs["token"] == "test-user-token"


@pytest.mark.asyncio
async def test_get_sdk_client_uses_x_datarobot_api_token() -> None:
    with (
        patch("datarobot.Client") as mock_client,
        patch("datarobot_genai.drmcp.core.clients.get_http_headers") as mock_get_headers,
        patch("datarobot_genai.drmcp.core.clients.get_credentials") as mock_get_creds,
    ):
        mock_get_headers.return_value = {"x-datarobot-api-token": "Bearer test-user-token"}
        mock_creds = MagicMock()
        mock_creds.datarobot.endpoint = "test-endpoint"
        mock_get_creds.return_value = mock_creds
        get_sdk_client()
        mock_client.assert_called_once()
        args, kwargs = mock_client.call_args
        assert kwargs["token"] == "test-user-token"


@pytest.mark.asyncio
async def test_get_sdk_client_raises_when_no_token_in_headers() -> None:
    """get_sdk_client raises when no token in headers (no credentials fallback)."""
    with (
        patch("datarobot_genai.drmcp.core.clients.get_http_headers") as mock_get_headers,
        patch("datarobot_genai.drmcp.core.clients.get_credentials") as mock_get_creds,
    ):
        mock_get_headers.return_value = {}
        mock_creds = MagicMock()
        mock_creds.datarobot.application_api_token = "env-token"
        mock_creds.datarobot.endpoint = "env-endpoint"
        mock_get_creds.return_value = mock_creds
        with pytest.raises(ValueError, match="No API token found"):
            get_sdk_client()


@pytest.mark.asyncio
async def test_get_sdk_client_uses_token_from_headers() -> None:
    """get_sdk_client uses token from headers when present."""
    with (
        patch("datarobot.Client") as mock_client,
        patch("datarobot_genai.drmcp.core.clients.get_http_headers") as mock_get_headers,
        patch("datarobot_genai.drmcp.core.clients.get_credentials") as mock_get_creds,
    ):
        mock_get_headers.return_value = {"authorization": "Bearer header-token"}
        mock_creds = MagicMock()
        mock_creds.datarobot.endpoint = "env-endpoint"
        mock_get_creds.return_value = mock_creds
        get_sdk_client()
        mock_client.assert_called_once()
        _, kwargs = mock_client.call_args
        assert kwargs["token"] == "header-token"
        assert kwargs["endpoint"] == "env-endpoint"
