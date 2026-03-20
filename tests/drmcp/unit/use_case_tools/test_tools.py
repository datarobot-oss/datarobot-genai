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
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drtools.use_case import tools


@pytest.mark.asyncio
async def test_list_use_cases_success() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"id": "uc1", "name": "Use Case 1"},
            {"id": "uc2", "name": "Use Case 2"},
        ]
    }
    mock_rest_client = MagicMock()
    mock_rest_client.get.return_value = mock_response
    mock_dr_module = MagicMock()
    mock_dr_module.client.get_client.return_value = mock_rest_client
    with (
        patch(
            "datarobot_genai.drtools.use_case.tools.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.use_case.tools.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_dr_module

        result = await tools.list_use_cases()
        assert isinstance(result, ToolResult)
        assert result.structured_content["count"] == 2
        assert result.structured_content["use_cases"][0]["id"] == "uc1"


@pytest.mark.asyncio
async def test_list_use_cases_with_search() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"id": "uc1", "name": "Matching Case"}]}
    mock_rest_client = MagicMock()
    mock_rest_client.get.return_value = mock_response
    mock_dr_module = MagicMock()
    mock_dr_module.client.get_client.return_value = mock_rest_client
    with (
        patch(
            "datarobot_genai.drtools.use_case.tools.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.use_case.tools.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_dr_module

        result = await tools.list_use_cases(search="Matching")
        mock_rest_client.get.assert_called_once_with(
            "useCases/", params={"limit": 100, "search": "Matching"}
        )
        assert result.structured_content["count"] == 1


@pytest.mark.asyncio
async def test_list_use_cases_empty() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": []}
    mock_rest_client = MagicMock()
    mock_rest_client.get.return_value = mock_response
    mock_dr_module = MagicMock()
    mock_dr_module.client.get_client.return_value = mock_rest_client
    with (
        patch(
            "datarobot_genai.drtools.use_case.tools.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.use_case.tools.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_dr_module

        result = await tools.list_use_cases()
        assert result.structured_content["count"] == 0
        assert result.structured_content["use_cases"] == []


@pytest.mark.asyncio
async def test_list_use_case_assets_success() -> None:
    with (
        patch(
            "datarobot_genai.drtools.use_case.tools.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.use_case.tools.DataRobotClient") as mock_drc,
    ):
        mock_client = MagicMock()
        mock_use_case = MagicMock()
        mock_use_case.name = "Test UC"

        mock_ds = MagicMock(id="ds1", name="Dataset 1")
        mock_use_case.list_datasets.return_value = [mock_ds]

        mock_dep = MagicMock(id="dep1", label="Deployment 1")
        mock_use_case.list_deployments.return_value = [mock_dep]

        mock_proj = MagicMock(id="proj1", project_name="Project 1")
        mock_use_case.list_projects.return_value = [mock_proj]

        mock_client.UseCase.get.return_value = mock_use_case
        mock_drc.return_value.get_client.return_value = mock_client

        result = await tools.list_use_case_assets(use_case_id="uc1")
        assert isinstance(result, ToolResult)
        assert result.structured_content["name"] == "Test UC"
        assert len(result.structured_content["datasets"]) == 1
        assert len(result.structured_content["deployments"]) == 1
        assert len(result.structured_content["experiments"]) == 1


@pytest.mark.asyncio
async def test_list_use_case_assets_multiple_ids() -> None:
    with (
        patch(
            "datarobot_genai.drtools.use_case.tools.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.use_case.tools.DataRobotClient") as mock_drc,
    ):
        mock_client = MagicMock()

        mock_uc1 = MagicMock()
        mock_uc1.name = "UC 1"
        mock_uc1.list_datasets.return_value = [MagicMock(id="ds1", name="DS 1")]
        mock_uc1.list_deployments.return_value = []
        mock_uc1.list_projects.return_value = []

        mock_uc2 = MagicMock()
        mock_uc2.name = "UC 2"
        mock_uc2.list_datasets.return_value = []
        mock_uc2.list_deployments.return_value = [MagicMock(id="dep1", label="Dep 1")]
        mock_uc2.list_projects.return_value = []

        mock_client.UseCase.get.side_effect = lambda uid: {"uc1": mock_uc1, "uc2": mock_uc2}[uid]
        mock_drc.return_value.get_client.return_value = mock_client

        result = await tools.list_use_case_assets(use_case_ids=["uc1", "uc2"])
        assert isinstance(result, ToolResult)
        assert result.structured_content["count"] == 2
        assert result.structured_content["use_cases"][0]["name"] == "UC 1"
        assert result.structured_content["use_cases"][1]["name"] == "UC 2"


@pytest.mark.asyncio
async def test_list_use_case_assets_missing_id() -> None:
    with pytest.raises(ToolError, match="use_case_id.*or.*use_case_ids.*must be provided"):
        await tools.list_use_case_assets()


@pytest.mark.asyncio
async def test_list_use_case_assets_partial_error() -> None:
    with (
        patch(
            "datarobot_genai.drtools.use_case.tools.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.use_case.tools.DataRobotClient") as mock_drc,
    ):
        mock_client = MagicMock()
        mock_use_case = MagicMock()
        mock_use_case.name = "Test UC"

        mock_use_case.list_datasets.side_effect = Exception("dataset error")
        mock_use_case.list_deployments.return_value = []
        mock_use_case.list_projects.return_value = []

        mock_client.UseCase.get.return_value = mock_use_case
        mock_drc.return_value.get_client.return_value = mock_client

        result = await tools.list_use_case_assets(use_case_id="uc1")
        assert "datasets_error" in result.structured_content
        assert result.structured_content["deployments"] == []
