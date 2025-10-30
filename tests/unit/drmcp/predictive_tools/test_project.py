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

from datarobot_genai.drmcp.tools.predictive import project


@pytest.mark.asyncio
async def test_list_projects_success() -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.project.get_sdk_client") as mock_get_client:
        mock_client = MagicMock()
        mock_proj1 = MagicMock(id="1", project_name="proj1")
        mock_proj2 = MagicMock(id="2", project_name="proj2")
        mock_client.Project.list.return_value = [mock_proj1, mock_proj2]
        mock_get_client.return_value = mock_client
        result = await project.list_projects()
        assert "1: proj1" in result
        assert "2: proj2" in result


@pytest.mark.asyncio
async def test_list_projects_empty() -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.project.get_sdk_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.Project.list.return_value = []
        mock_get_client.return_value = mock_client
        result = await project.list_projects()
        assert "No projects found." in result


@pytest.mark.asyncio
async def test_list_projects_error() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.project.get_sdk_client",
        side_effect=Exception("fail"),
    ):
        with pytest.raises(Exception) as exc_info:
            await project.list_projects()
        assert "Error in list_projects: Exception: fail" == str(exc_info.value)


@pytest.mark.asyncio
async def test_get_project_dataset_by_name_success() -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.project.get_sdk_client") as mock_get_client:
        mock_client = MagicMock()
        mock_project = MagicMock()
        mock_ds1 = MagicMock()
        mock_ds1.name = "training_data"
        mock_ds1.id = "dsid"
        mock_project.get_datasets.return_value = [mock_ds1]
        mock_client.Project.get.return_value = mock_project
        mock_get_client.return_value = mock_client
        result = await project.get_project_dataset_by_name("pid", "training")
        mock_client.Project.get.assert_called_once_with("pid")
        assert "dsid" in result


@pytest.mark.asyncio
async def test_get_project_dataset_by_name_not_found() -> None:
    with patch("datarobot_genai.drmcp.tools.predictive.project.get_sdk_client") as mock_get_client:
        mock_client = MagicMock()
        mock_project = MagicMock()
        mock_project.get_datasets.return_value = []
        mock_client.Project.get.return_value = mock_project
        mock_get_client.return_value = mock_client
        result = await project.get_project_dataset_by_name("pid", "training")
        assert "Dataset with name containing 'training' not found in project pid." in result


@pytest.mark.asyncio
async def test_get_project_dataset_by_name_error() -> None:
    with patch(
        "datarobot_genai.drmcp.tools.predictive.project.get_sdk_client",
        side_effect=Exception("fail"),
    ):
        with pytest.raises(Exception) as exc_info:
            await project.get_project_dataset_by_name("pid", "training")
        assert "Error in get_project_dataset_by_name: Exception: fail" == str(exc_info.value)
