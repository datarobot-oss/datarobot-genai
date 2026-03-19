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
    use_case_id: Annotated[str, "The ID of a single DataRobot use case"] | None = None,
    use_case_ids: Annotated[list[str], "List of use case IDs to fetch assets for"] | None = None,
) -> ToolError | ToolResult:
    """List datasets, deployments, and experiments belonging to one or more use cases."""
    multi = use_case_ids is not None
    ids: list[str] = []
    if use_case_ids is not None:
        if not use_case_ids:
            raise ToolError("use_case_ids must not be empty.")
        ids = use_case_ids
    elif use_case_id:
        ids = [use_case_id]
    else:
        raise ToolError("Either use_case_id or use_case_ids must be provided.")

    token = await get_datarobot_access_token()
    dr_module = DataRobotClient(token).get_client()

    results: list[dict] = []
    for uc_id in ids:
        entry: dict = {"use_case_id": uc_id}

        try:
            use_case = dr_module.UseCase.get(uc_id)
        except Exception as exc:
            entry["error"] = str(exc)
            results.append(entry)
            continue

        entry["name"] = use_case.name

        try:
            datasets = list(use_case.list_datasets())
            entry["datasets"] = [{"id": d.id, "name": d.name} for d in datasets]
        except Exception as exc:
            entry["datasets_error"] = str(exc)

        try:
            deployments = list(use_case.list_deployments())
            entry["deployments"] = [{"id": d.id, "label": d.label} for d in deployments]
        except Exception as exc:
            entry["deployments_error"] = str(exc)

        try:
            projects = list(use_case.list_projects())
            entry["experiments"] = [{"id": p.id, "name": p.project_name} for p in projects]
        except Exception as exc:
            entry["experiments_error"] = str(exc)

        results.append(entry)

    # use_case_id → flat dict; use_case_ids → always list format
    if not multi:
        return ToolResult(structured_content=results[0])
    return ToolResult(structured_content={"use_cases": results, "count": len(results)})
