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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.predictive import project


@pytest.mark.asyncio
async def test_list_projects_success() -> None:
    mock_client = MagicMock()
    mock_proj1 = MagicMock(id="1", project_name="proj1")
    mock_proj2 = MagicMock(id="2", project_name="proj2")
    mock_client.Project.list.return_value = [mock_proj1, mock_proj2]
    with (
        patch(
            "datarobot_genai.drtools.predictive.project.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.project.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        result = await project.list_projects()
    assert isinstance(result, dict)
    projects_dict = result
    assert "1" in projects_dict
    assert projects_dict["1"] == "proj1"
    assert "2" in projects_dict
    assert projects_dict["2"] == "proj2"


@pytest.mark.asyncio
async def test_list_projects_paged_success() -> None:
    mock_client = MagicMock()
    mock_proj = MagicMock(id="p1", project_name="one")
    mock_client.Project.list.return_value = [mock_proj]
    with (
        patch(
            "datarobot_genai.drtools.predictive.project.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.project.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        result = await project.list_projects(offset=10, limit=5)
    mock_client.Project.list.assert_called_once_with(offset=10, limit=5)
    assert result["projects"]["p1"] == "one"
    assert result["offset"] == 10
    assert result["limit"] == 5
    assert result["returned_count"] == 1


@pytest.mark.asyncio
async def test_list_projects_empty() -> None:
    mock_client = MagicMock()
    mock_client.Project.list.return_value = []
    with (
        patch(
            "datarobot_genai.drtools.predictive.project.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.project.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        result = await project.list_projects()
    assert isinstance(result, dict)
    assert result == {}


@pytest.mark.asyncio
async def test_list_projects_error() -> None:
    with patch(
        "datarobot_genai.drtools.predictive.project.get_datarobot_access_token",
        new_callable=AsyncMock,
        side_effect=Exception("fail"),
    ):
        with pytest.raises(Exception) as exc_info:
            await project.list_projects()
        assert "fail" == str(exc_info.value)


@pytest.mark.asyncio
async def test_get_project_dataset_by_name_success() -> None:
    mock_client = MagicMock()
    mock_project = MagicMock()
    mock_ds1 = MagicMock()
    mock_ds1.name = "training_data"
    mock_ds1.id = "dsid"
    mock_project.get_datasets.return_value = [mock_ds1]
    mock_project.get_dataset.return_value = None
    mock_client.Project.get.return_value = mock_project
    with (
        patch(
            "datarobot_genai.drtools.predictive.project.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.project.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        result = await project.get_project_dataset_by_name(
            project_id="pid", dataset_name="training"
        )
    mock_client.Project.get.assert_called_once_with("pid")
    assert isinstance(result, dict)
    assert result["dataset_id"] == "dsid"


@pytest.mark.asyncio
async def test_get_project_dataset_by_name_not_found() -> None:
    mock_client = MagicMock()
    mock_project = MagicMock()
    mock_project.get_datasets.return_value = []
    mock_project.get_dataset.return_value = None
    mock_client.Project.get.return_value = mock_project
    with (
        patch(
            "datarobot_genai.drtools.predictive.project.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.predictive.project.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_client
        with pytest.raises(ToolError) as exc_info:
            await project.get_project_dataset_by_name(project_id="pid", dataset_name="training")
    assert (
        str(exc_info.value) == "Dataset with name containing 'training' not found in project pid."
    )


@pytest.mark.asyncio
async def test_get_project_dataset_by_name_error() -> None:
    with patch(
        "datarobot_genai.drtools.predictive.project.get_datarobot_access_token",
        new_callable=AsyncMock,
        side_effect=Exception("fail"),
    ):
        with pytest.raises(Exception) as exc_info:
            await project.get_project_dataset_by_name(project_id="pid", dataset_name="training")
        assert "fail" == str(exc_info.value)
