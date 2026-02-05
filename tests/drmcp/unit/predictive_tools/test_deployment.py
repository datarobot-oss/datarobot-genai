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

import os
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from fastmcp.exceptions import ToolError

from datarobot_genai.drmcp.tools.predictive import deployment
from datarobot_genai.drmcp.tools.predictive.custom_model_deploy import (
    deploy_custom_model_impl,
)


def test_load_dotenv() -> None:
    load_dotenv(verbose=True)


@pytest.mark.asyncio
async def test_list_deployments_success() -> None:
    mock_client = MagicMock()
    mock_dep1 = MagicMock(id="1", label="dep1")
    mock_dep2 = MagicMock(id="2", label="dep2")
    mock_client.Deployment.list.return_value = [mock_dep1, mock_dep2]
    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.deployment.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        result = await deployment.list_deployments()
        assert result.structured_content["deployments"]["1"] == "dep1"
        assert result.structured_content["deployments"]["2"] == "dep2"


@pytest.mark.asyncio
async def test_list_deployments_empty() -> None:
    mock_client = MagicMock()
    mock_client.Deployment.list.return_value = []
    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.deployment.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        result = await deployment.list_deployments()
        assert result.structured_content["deployments"] == []


@pytest.mark.asyncio
async def test_list_deployments_error() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.deployment.get_datarobot_access_token",
        new_callable=AsyncMock,
        side_effect=Exception("fail"),
    ):
        with pytest.raises(ToolError) as exc_info:
            await deployment.list_deployments()
        assert "Error in list_deployments: Exception: fail" == str(exc_info.value)


@pytest.mark.asyncio
async def test_get_model_info_from_deployment_success() -> None:
    mock_client = MagicMock()
    mock_deployment = MagicMock()
    mock_deployment.model = {"project_id": "pid", "model_id": "mid"}
    mock_client.Deployment.get.return_value = mock_deployment
    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.deployment.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        result = await deployment.get_model_info_from_deployment(deployment_id="dep_id")
        mock_client.Deployment.get.assert_called_once_with("dep_id")
        assert "project_id" in result.content[0].text
        assert result.structured_content["project_id"] == "pid"


@pytest.mark.asyncio
async def test_get_model_info_from_deployment_not_found() -> None:
    mock_client = MagicMock()
    mock_client.Deployment.get.side_effect = Exception("404 client error: {'message': 'Not Found'}")
    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.deployment.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(ToolError) as exc_info:
            await deployment.get_model_info_from_deployment(deployment_id="dep_id")
        assert (
            "Error in get_model_info_from_deployment: Exception: 404 client error: "
            "{'message': 'Not Found'}" == str(exc_info.value)
        )


@pytest.mark.asyncio
async def test_get_model_info_from_deployment_error() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.deployment.get_datarobot_access_token",
        new_callable=AsyncMock,
        side_effect=Exception("fail"),
    ):
        with pytest.raises(ToolError) as exc_info:
            await deployment.get_model_info_from_deployment(deployment_id="dep_id")
        assert "Error in get_model_info_from_deployment: Exception: fail" == str(exc_info.value)


@pytest.mark.asyncio
async def test_deploy_model_success() -> None:
    mock_client = MagicMock()
    mock_server = MagicMock(id="srv1")
    mock_client.PredictionServer.list.return_value = [mock_server]
    mock_deployment = MagicMock(id="dep123")
    mock_client.Deployment.create_from_learning_model.return_value = mock_deployment
    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.deployment.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        result = await deployment.deploy_model(
            model_id="model123", label="Test Deployment", description="desc"
        )
        assert result.structured_content["deployment_id"] == "dep123"
        assert result.structured_content["label"] == "Test Deployment"


@pytest.mark.asyncio
async def test_deploy_model_no_prediction_servers() -> None:
    mock_client = MagicMock()
    mock_client.PredictionServer.list.return_value = []
    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.deployment.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(ToolError) as exc_info:
            await deployment.deploy_model(model_id="model123", label="Test Deployment")
        assert "No prediction servers available" in str(exc_info.value)


@pytest.mark.asyncio
async def test_deploy_model_error() -> None:
    mock_client = MagicMock()
    mock_client.PredictionServer.list.side_effect = Exception("fail servers")
    with (
        patch(
            "datarobot_genai.drmcp.tools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drmcp.tools.predictive.deployment.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(ToolError) as exc_info:
            await deployment.deploy_model(model_id="model123", label="Test Deployment")
        assert "fail servers" in str(exc_info.value)


def _custom_model_fixture_dir() -> str:
    return str(Path(__file__).resolve().parent.parent.parent / "fixtures" / "custom_model")


@pytest.mark.asyncio
async def test_deploy_custom_model_validation_missing_model_folder() -> None:
    with pytest.raises(ToolError) as exc_info:
        await deployment.deploy_custom_model(
            name="x", target_type="Binary", target_name="t"
        )
    assert "model_folder" in str(exc_info.value).lower() or "missing" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_deploy_custom_model_validation_missing_name() -> None:
    folder = _custom_model_fixture_dir()
    with pytest.raises(ToolError) as exc_info:
        await deployment.deploy_custom_model(
            model_folder=folder, target_type="Binary", target_name="t"
        )
    assert "name" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_deploy_custom_model_validation_folder_not_directory() -> None:
    with pytest.raises(ToolError) as exc_info:
        await deployment.deploy_custom_model(
            model_folder="/nonexistent_path_12345",
            name="x",
            target_type="Binary",
            target_name="t",
        )
    assert "not a directory" in str(exc_info.value) or "nonexistent" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_deploy_custom_model_mocked_success() -> None:
    folder = _custom_model_fixture_dir()
    model_file = os.path.join(folder, "custom.py")
    out = {"deployment_id": "dep1", "label": "Test", "custom_model_id": "cm1", "custom_model_version_id": "v1", "registered_model_version_id": "rmv1"}
    with patch(
        "datarobot_genai.drmcp.tools.predictive.deployment.deploy_custom_model_impl",
        return_value=out,
    ):
        result = await deployment.deploy_custom_model(
            model_folder=folder,
            model_file_path=model_file,
            name="Test",
            target_type="Binary",
            target_name="target",
        )
    assert result.structured_content["deployment_id"] == "dep1"
    assert result.structured_content["label"] == "Test"
    assert result.structured_content["custom_model_id"] == "cm1"


@pytest.mark.asyncio
async def test_deploy_custom_model_no_model_file_raises_tool_error() -> None:
    folder = _custom_model_fixture_dir()
    with pytest.raises(ToolError) as exc_info:
        await deployment.deploy_custom_model(
            model_folder=folder,
            name="Test",
            target_type="Binary",
            target_name="target",
        )
    assert "model file" in str(exc_info.value).lower()
    assert "model_file_path" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_deploy_custom_model_no_model_file_with_model_file_path_succeeds() -> None:
    folder = _custom_model_fixture_dir()
    provided_path = os.path.join(folder, "custom.py")
    out = {"deployment_id": "d", "label": "L"}
    with patch(
        "datarobot_genai.drmcp.tools.predictive.deployment.deploy_custom_model_impl",
        return_value=out,
    ):
        result = await deployment.deploy_custom_model(
            model_folder=folder,
            model_file_path=provided_path,
            name="Test",
            target_type="Binary",
            target_name="target",
        )
    assert result.structured_content["deployment_id"] == "d"


@pytest.mark.asyncio
async def test_deploy_custom_model_impl_no_prediction_servers_raises_error() -> None:
    """Test that deploy_custom_model_impl raises an error when no prediction servers are available.
    
    This test verifies the issue where deploy_custom_model_impl doesn't validate prediction servers
    before attempting deployment, unlike deploy_model which does validate.
    """
    folder = _custom_model_fixture_dir()
    model_file = os.path.join(folder, "model.pkl")
    # Create a dummy model file for testing
    os.makedirs(folder, exist_ok=True)
    with open(model_file, "wb") as f:
        f.write(b"dummy model data")
    
    try:
        with patch(
            "datarobot_genai.drmcp.tools.predictive.custom_model_deploy.get_sdk_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            
            # Mock execution environment - needs to match the pattern in select_execution_environment
            mock_env = MagicMock()
            mock_env.id = "env123"
            mock_env.name = "[DataRobot] Python Scikit-Learn 3.11 Drop-In"
            mock_env.latest_successful_version = MagicMock()
            mock_env.latest_successful_version.id = "env_ver123"
            mock_client.ExecutionEnvironment.list.return_value = [mock_env]
            
            # Mock custom model creation
            mock_custom_model = MagicMock()
            mock_custom_model.id = "cm123"
            mock_client.CustomInferenceModel.create.return_value = mock_custom_model
            
            # Mock custom model version
            mock_version = MagicMock()
            mock_version.id = "version123"
            mock_client.CustomModelVersion.create_clean.return_value = mock_version
            
            # Mock dependency build
            mock_build_info = MagicMock()
            mock_build_info.build_status = "success"
            mock_client.CustomModelVersionDependencyBuild.start_build.return_value = mock_build_info
            
            # Mock registered model version
            mock_rmv = MagicMock()
            mock_rmv.id = "rmv123"
            mock_client.RegisteredModelVersion.create_for_custom_model_version.return_value = mock_rmv
            
            # Mock prediction servers - EMPTY LIST (this is the issue scenario)
            mock_client.PredictionServer.list.return_value = []
            
            # Mock deployment creation - this should fail if ps_id is None
            mock_deployment = MagicMock()
            mock_deployment.id = "dep123"
            mock_deployment.label = "Test Model"
            mock_client.Deployment.create_from_registered_model_version.return_value = mock_deployment
            
            mock_get_client.return_value = mock_client
            
            # This should raise a ValueError when no prediction servers exist
            # This ensures consistency with deploy_model which also validates prediction servers
            with pytest.raises((ValueError, RuntimeError)) as exc_info:
                deploy_custom_model_impl(
                    model_folder=folder,
                    model_file_path=model_file,
                    name="Test Model",
                    target_type="Binary",
                    target_name="target",
                )
            # Verify the error message mentions prediction servers
            assert "prediction server" in str(exc_info.value).lower() or "no prediction" in str(exc_info.value).lower()
    finally:
        # Cleanup
        if os.path.exists(model_file):
            os.remove(model_file)
