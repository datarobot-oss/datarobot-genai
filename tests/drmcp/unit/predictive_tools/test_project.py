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

import datarobot as dr
import pytest

from datarobot_genai.drtools.core.clients.datarobot import ThreadSafeDataRobotClient
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.predictive import project


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_list_projects_success() -> None:
    mock_proj1 = MagicMock(id="1", project_name="proj1")
    mock_proj2 = MagicMock(id="2", project_name="proj2")
    with patch.object(dr.Project, "list", return_value=[mock_proj1, mock_proj2]):
        result = await project.list_projects()
    assert isinstance(result, dict)
    projects_dict = result
    assert "1" in projects_dict
    assert projects_dict["1"] == "proj1"
    assert "2" in projects_dict
    assert projects_dict["2"] == "proj2"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_list_projects_empty() -> None:
    with patch.object(dr.Project, "list", return_value=[]):
        result = await project.list_projects()
    assert isinstance(result, dict)
    assert result == {}


@pytest.mark.asyncio
async def test_list_projects_error() -> None:
    with patch.object(
        ThreadSafeDataRobotClient,
        "get_client_context_with_token_from_request_header",
        side_effect=Exception("fail"),
    ):
        with pytest.raises(Exception) as exc_info:
            await project.list_projects()
        assert "fail" == str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_get_project_dataset_by_name_success() -> None:
    mock_project = MagicMock()
    mock_ds1 = MagicMock()
    mock_ds1.name = "training_data"
    mock_ds1.id = "dsid"
    mock_project.get_datasets.return_value = [mock_ds1]
    mock_project.get_dataset.return_value = None
    with patch.object(dr.Project, "get", return_value=mock_project) as mock_proj_get:
        result = await project.get_project_dataset_by_name(
            project_id="pid", dataset_name="training"
        )
    mock_proj_get.assert_called_once_with("pid")
    assert isinstance(result, dict)
    assert result["dataset_id"] == "dsid"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_get_client_context_with_token_from_request_header")
async def test_get_project_dataset_by_name_not_found() -> None:
    mock_project = MagicMock()
    mock_project.get_datasets.return_value = []
    mock_project.get_dataset.return_value = None
    with patch.object(dr.Project, "get", return_value=mock_project):
        with pytest.raises(ToolError) as exc_info:
            await project.get_project_dataset_by_name(project_id="pid", dataset_name="training")
    assert (
        str(exc_info.value) == "Dataset with name containing 'training' not found in project pid."
    )


@pytest.mark.asyncio
async def test_get_project_dataset_by_name_error() -> None:
    with patch.object(
        ThreadSafeDataRobotClient,
        "get_client_context_with_token_from_request_header",
        side_effect=Exception("fail"),
    ):
        with pytest.raises(Exception) as exc_info:
            await project.get_project_dataset_by_name(project_id="pid", dataset_name="training")
        assert "fail" == str(exc_info.value)
