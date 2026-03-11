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
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drtools.optimization import tools


@pytest.mark.asyncio
async def test_cuopt_solve_success() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "status": "optimal",
                "objective_value": 42.0,
                "solution": {"x": [1, 2, 3]},
                "solver_info": {"iterations": 100},
            }
        ]
    }
    with (
        patch.dict(os.environ, {"CUOPT_DEPLOYMENT_ID": "dep1"}),
        patch(
            "datarobot_genai.drtools.optimization.tools.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.optimization.tools.DataRobotClient") as mock_drc,
    ):
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_drc.return_value.get_client.return_value = mock_client

        result = await tools.cuopt_solve(
            problem_definition={"objective": "minimize", "constraints": []}
        )
        assert isinstance(result, ToolResult)
        assert result.structured_content["status"] == "optimal"
        assert result.structured_content["objective_value"] == 42.0
        assert result.structured_content["preview"] is False


@pytest.mark.asyncio
async def test_cuopt_solve_preview() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"valid": True}
    with (
        patch.dict(os.environ, {"CUOPT_DEPLOYMENT_ID": "dep1"}),
        patch(
            "datarobot_genai.drtools.optimization.tools.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.optimization.tools.DataRobotClient") as mock_drc,
    ):
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_drc.return_value.get_client.return_value = mock_client

        result = await tools.cuopt_solve(
            problem_definition={"objective": "minimize"}, preview=True
        )
        assert isinstance(result, ToolResult)
        assert result.structured_content["preview"] is True


@pytest.mark.asyncio
async def test_cuopt_solve_missing_deployment_id() -> None:
    with patch.dict(os.environ, {}, clear=False):
        # Ensure CUOPT_DEPLOYMENT_ID is not set
        os.environ.pop("CUOPT_DEPLOYMENT_ID", None)
        with pytest.raises(ToolError, match="CUOPT_DEPLOYMENT_ID not configured"):
            await tools.cuopt_solve(
                problem_definition={"objective": "minimize"}
            )


@pytest.mark.asyncio
async def test_cuopt_solve_missing_problem_definition() -> None:
    with pytest.raises(ToolError, match="Problem definition must be provided"):
        await tools.cuopt_solve()


@pytest.mark.asyncio
async def test_cuopt_solve_no_solution() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": []}
    with (
        patch.dict(os.environ, {"CUOPT_DEPLOYMENT_ID": "dep1"}),
        patch(
            "datarobot_genai.drtools.optimization.tools.get_datarobot_access_token",
            new_callable=AsyncMock,
            return_value="token",
        ),
        patch("datarobot_genai.drtools.optimization.tools.DataRobotClient") as mock_drc,
    ):
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_drc.return_value.get_client.return_value = mock_client

        with pytest.raises(ToolError, match="No solution returned"):
            await tools.cuopt_solve(
                problem_definition={"objective": "minimize"}
            )
