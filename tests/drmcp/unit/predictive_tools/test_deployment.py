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
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from fastmcp.exceptions import ToolError

from datarobot_genai.drtools.clients.datarobot import deploy_custom_model_impl
from datarobot_genai.drtools.predictive import deployment


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
            "datarobot_genai.drtools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.deployment.DataRobotClient") as mock_drc,
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
            "datarobot_genai.drtools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.deployment.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        result = await deployment.list_deployments()
        assert result.structured_content["deployments"] == []


@pytest.mark.asyncio
async def test_list_deployments_error() -> None:
    with patch(
        "datarobot_genai.drtools.predictive.deployment.get_datarobot_access_token",
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
            "datarobot_genai.drtools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.deployment.DataRobotClient") as mock_drc,
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
            "datarobot_genai.drtools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.deployment.DataRobotClient") as mock_drc,
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
        "datarobot_genai.drtools.predictive.deployment.get_datarobot_access_token",
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
            "datarobot_genai.drtools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.deployment.DataRobotClient") as mock_drc,
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
            "datarobot_genai.drtools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.deployment.DataRobotClient") as mock_drc,
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
            "datarobot_genai.drtools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.deployment.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(ToolError) as exc_info:
            await deployment.deploy_model(model_id="model123", label="Test Deployment")
        assert "fail servers" in str(exc_info.value)


# custom_model fixtures: custom.py and requirements.txt are required by the DataRobot
# custom model format, not by pytest. They are used as fixture content when testing
# deploy_custom_model.
def _custom_model_fixture_dir() -> str:
    return str(Path(__file__).resolve().parent / "fixtures" / "custom_model")


@pytest.mark.asyncio
async def test_deploy_custom_model_validation_missing_model_folder() -> None:
    with pytest.raises(ToolError) as exc_info:
        await deployment.deploy_custom_model(name="x", target_type="Binary", target_name="t")
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
    out = {
        "deployment_id": "dep1",
        "label": "Test",
        "custom_model_id": "cm1",
        "custom_model_version_id": "v1",
        "registered_model_version_id": "rmv1",
    }
    with (
        patch(
            "datarobot_genai.drtools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.deployment.DataRobotClient") as mock_drc,
        patch(
            "datarobot_genai.drtools.predictive.deployment.deploy_custom_model_impl",
            return_value=out,
        ),
    ):
        mock_drc.return_value.get_client.return_value = MagicMock()
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
    fixture_dir = Path(_custom_model_fixture_dir())
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in ("custom.py", "requirements.txt"):
            shutil.copy(fixture_dir / f, os.path.join(tmpdir, f))
        with pytest.raises(ToolError) as exc_info:
            await deployment.deploy_custom_model(
                model_folder=tmpdir,
                name="Test",
                target_type="Binary",
                target_name="target",
            )
        err = str(exc_info.value).lower()
        assert "model file" in err
        assert "model_file_path" in err


@pytest.mark.asyncio
async def test_deploy_custom_model_explicit_model_file_path_nonexistent_raises_error() -> None:
    """model_file_path given but file missing: raise ToolError (no silent fallback)."""
    fixture_dir = Path(_custom_model_fixture_dir())
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in ("custom.py", "requirements.txt", "model.pkl"):
            src = fixture_dir / f
            if src.exists():
                shutil.copy(src, os.path.join(tmpdir, f))
        model_pkl = os.path.join(tmpdir, "model.pkl")
        if not os.path.exists(model_pkl):
            with open(model_pkl, "wb") as f:
                f.write(b"dummy")
        nonexistent = os.path.join(tmpdir, "nonexistent_model.pkl")
        assert not os.path.exists(nonexistent)
        with pytest.raises(ToolError) as exc_info:
            await deployment.deploy_custom_model(
                model_folder=tmpdir,
                model_file_path=nonexistent,
                name="Test",
                target_type="Binary",
                target_name="target",
            )
        err = str(exc_info.value)
        assert "does not exist" in err
        assert nonexistent in err or "nonexistent" in err.lower()


@pytest.mark.asyncio
async def test_deploy_custom_model_no_model_file_with_model_file_path_succeeds() -> None:
    folder = _custom_model_fixture_dir()
    provided_path = os.path.join(folder, "custom.py")
    out = {"deployment_id": "d", "label": "L"}
    with (
        patch(
            "datarobot_genai.drtools.predictive.deployment.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.deployment.DataRobotClient") as mock_drc,
        patch(
            "datarobot_genai.drtools.predictive.deployment.deploy_custom_model_impl",
            return_value=out,
        ),
    ):
        mock_drc.return_value.get_client.return_value = MagicMock()
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
    """deploy_custom_model_impl checks prediction servers first; no resources created when none."""
    fixture_dir = Path(_custom_model_fixture_dir())
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in ("custom.py", "requirements.txt", "model.pkl"):
            src = fixture_dir / f
            if src.exists():
                shutil.copy(src, os.path.join(tmpdir, f))
        model_file = os.path.join(tmpdir, "model.pkl")
        if not os.path.exists(model_file):
            with open(model_file, "wb") as f:
                f.write(b"dummy model data")
        mock_client = MagicMock()
        mock_client.PredictionServer.list.return_value = []

        with pytest.raises(ValueError) as exc_info:
            deploy_custom_model_impl(
                mock_client,
                model_folder=tmpdir,
                model_file_path=model_file,
                name="Test Model",
                target_type="Binary",
                target_name="target",
            )
        assert (
            "prediction server" in str(exc_info.value).lower()
            or "no prediction" in str(exc_info.value).lower()
        )
        mock_client.CustomInferenceModel.create.assert_not_called()
