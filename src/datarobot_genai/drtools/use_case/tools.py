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

"""DataRobot use case tools."""

import logging
from typing import Annotated

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp import dr_mcp_integration_tool
from datarobot_genai.drtools.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.clients.datarobot import get_datarobot_access_token

logger = logging.getLogger(__name__)


@dr_mcp_integration_tool(tags={"use_case", "read", "list", "daria"})
async def list_use_cases(
    *,
    search: Annotated[str, "Optional search filter for use case names"] | None = None,
    limit: Annotated[int, "Maximum number of use cases to return"] = 100,
) -> ToolResult:
    """List DataRobot use cases with optional search filter."""
    token = await get_datarobot_access_token()
    dr_module = DataRobotClient(token).get_client()
    rest_client = dr_module.client.get_client()

    params: dict = {"limit": limit}
    if search:
        params["search"] = search
    response = rest_client.get("useCases/", params=params)
    items = response.json().get("data", [])

    return ToolResult(
        structured_content={
            "use_cases": [
                {"id": item["id"], "name": item.get("name", "Untitled")} for item in items
            ],
            "count": len(items),
        },
    )


@dr_mcp_integration_tool(tags={"use_case", "read", "assets", "daria"})
async def list_use_case_assets(
    *,
    use_case_id: Annotated[str, "The ID of the DataRobot use case"] | None = None,
) -> ToolError | ToolResult:
    """List datasets, deployments, and experiments belonging to a use case."""
    if not use_case_id:
        raise ToolError("Use case ID must be provided")

    token = await get_datarobot_access_token()
    dr_module = DataRobotClient(token).get_client()
    use_case = dr_module.UseCase.get(use_case_id)

    result: dict = {
        "use_case_id": use_case_id,
        "name": use_case.name,
    }

    try:
        datasets = list(use_case.list_datasets())
        result["datasets"] = [{"id": d.id, "name": d.name} for d in datasets]
    except Exception as exc:
        result["datasets_error"] = str(exc)

    try:
        deployments = list(use_case.list_deployments())
        result["deployments"] = [{"id": d.id, "label": d.label} for d in deployments]
    except Exception as exc:
        result["deployments_error"] = str(exc)

    try:
        projects = list(use_case.list_projects())
        result["experiments"] = [{"id": p.id, "name": p.project_name} for p in projects]
    except Exception as exc:
        result["experiments_error"] = str(exc)

    return ToolResult(structured_content=result)
