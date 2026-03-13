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

from datarobot_genai.drtools.vdb import tools


@pytest.mark.asyncio
async def test_list_vector_databases_success() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "id": "dep1",
                "label": "VDB 1",
                "status": "active",
                "capabilities": {"supportsVectorDatabaseQuerying": True},
                "model": {},
            },
            {
                "id": "dep2",
                "label": "Non-VDB",
                "status": "active",
                "capabilities": {},
                "model": {},
            },
        ]
    }
    mock_rest_client = MagicMock()
    mock_rest_client.get.return_value = mock_response
    mock_dr_module = MagicMock()
    mock_dr_module.client.get_client.return_value = mock_rest_client
    with (
        patch(
            "datarobot_genai.drtools.vdb.tools.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.vdb.tools.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_dr_module

        result = await tools.list_vector_databases()
        assert isinstance(result, ToolResult)
        assert result.structured_content["count"] == 1
        assert result.structured_content["vector_databases"][0]["deployment_id"] == "dep1"


@pytest.mark.asyncio
async def test_list_vector_databases_empty() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": []}
    mock_rest_client = MagicMock()
    mock_rest_client.get.return_value = mock_response
    mock_dr_module = MagicMock()
    mock_dr_module.client.get_client.return_value = mock_rest_client
    with (
        patch(
            "datarobot_genai.drtools.vdb.tools.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.vdb.tools.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_dr_module

        result = await tools.list_vector_databases()
        assert result.structured_content["count"] == 0
        assert result.structured_content["vector_databases"] == []


@pytest.mark.asyncio
async def test_query_vector_database_success() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"content": "doc1", "metadata": {}}]}
    mock_rest_client = MagicMock()
    mock_rest_client.post.return_value = mock_response
    mock_dr_module = MagicMock()
    mock_dr_module.client.get_client.return_value = mock_rest_client
    mock_deployment = MagicMock(id="dep1")
    mock_dr_module.Deployment.get.return_value = mock_deployment
    with (
        patch(
            "datarobot_genai.drtools.vdb.tools.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.vdb.tools.DataRobotClient") as mock_drc,
    ):
        mock_drc.return_value.get_client.return_value = mock_dr_module

        result = await tools.query_vector_database(deployment_id="dep1", query="test query")
        assert isinstance(result, ToolResult)
        assert result.structured_content["count"] == 1
        assert result.structured_content["deployment_id"] == "dep1"


@pytest.mark.asyncio
async def test_query_vector_database_missing_deployment_id() -> None:
    with pytest.raises(ToolError, match="Deployment ID must be provided"):
        await tools.query_vector_database(query="test")


@pytest.mark.asyncio
async def test_query_vector_database_missing_query() -> None:
    with pytest.raises(ToolError, match="Query must be provided"):
        await tools.query_vector_database(deployment_id="dep1")
