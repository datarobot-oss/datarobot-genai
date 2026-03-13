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

"""NVIDIA cuOpt optimization tool."""

import logging
import os
from typing import Annotated
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp import dr_mcp_integration_tool
from datarobot_genai.drtools.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.clients.datarobot import get_datarobot_access_token

logger = logging.getLogger(__name__)


@dr_mcp_integration_tool(tags={"optimization", "cuopt", "solver", "daria"})
async def cuopt_solve(
    *,
    problem_definition: Annotated[dict[str, Any], "The optimization problem definition"]
    | None = None,
    preview: Annotated[bool, "Validate without solving when True"] = False,
) -> ToolError | ToolResult:
    """Solve LP, MILP, or VRP optimization problems using NVIDIA cuOpt.

    Requires CUOPT_DEPLOYMENT_ID environment variable pointing to a
    DataRobot deployment running the cuOpt solver.
    Set preview=True to validate the problem definition without solving.
    """
    if not problem_definition:
        raise ToolError("Problem definition must be provided")

    deployment_id = os.environ.get("CUOPT_DEPLOYMENT_ID")
    if not deployment_id:
        raise ToolError(
            "CUOPT_DEPLOYMENT_ID not configured. "
            "Set this environment variable to the DataRobot deployment ID "
            "running the cuOpt solver."
        )

    token = await get_datarobot_access_token()
    dr_module = DataRobotClient(token).get_client()
    rest_client = dr_module.client.get_client()

    if preview:
        response = rest_client.post(
            f"deployments/{deployment_id}/predictions/",
            json={"data": [{"problem": problem_definition, "mode": "validate"}]},
        )
        data = response.json()
        return ToolResult(
            structured_content={"preview": True, "validation": data},
        )

    response = rest_client.post(
        f"deployments/{deployment_id}/predictions/",
        json={"data": [{"problem": problem_definition, "mode": "solve"}]},
    )
    data = response.json()
    predictions = data.get("data", [])
    if not predictions:
        raise ToolError("No solution returned from cuOpt deployment")

    result = predictions[0]
    return ToolResult(
        structured_content={
            "preview": False,
            "status": result.get("status"),
            "objective_value": result.get("objective_value"),
            "solution": result.get("solution"),
            "solver_info": result.get("solver_info"),
        },
    )
