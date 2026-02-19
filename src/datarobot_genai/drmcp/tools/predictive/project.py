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

import logging
from typing import Annotated

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp import dr_mcp_integration_tool
from datarobot_genai.drmcp.tools.clients.datarobot import DataRobotClient
from datarobot_genai.drmcp.tools.clients.datarobot import get_datarobot_access_token

logger = logging.getLogger(__name__)


@dr_mcp_integration_tool(tags={"predictive", "project", "read", "management", "list"})
async def list_projects() -> ToolResult:
    """List all DataRobot projects for the authenticated user."""
    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    projects = client.Project.list()
    projects = {p.id: p.project_name for p in projects}

    return ToolResult(
        structured_content=projects,
    )


@dr_mcp_integration_tool(tags={"predictive", "project", "read", "data", "info"})
async def get_project_dataset_by_name(
    *,
    project_id: Annotated[str, "The ID of the DataRobot project."] | None = None,
    dataset_name: Annotated[str, "The name of the dataset to find (e.g., 'training', 'holdout')."]
    | None = None,
) -> ToolResult:
    """Get a dataset ID by name for a given project.

    The dataset ID and the dataset type (source or prediction) as a string, or an error message.
    """
    if not project_id:
        raise ToolError("Project ID is required.")
    if not dataset_name:
        raise ToolError("Dataset name is required.")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    project = client.Project.get(project_id)
    all_datasets = []
    source_dataset = project.get_dataset()
    if source_dataset:
        all_datasets.append({"type": "source", "dataset": source_dataset})
    prediction_datasets = project.get_datasets()
    if prediction_datasets:
        all_datasets.extend([{"type": "prediction", "dataset": ds} for ds in prediction_datasets])
    for ds in all_datasets:
        if dataset_name.lower() in ds["dataset"].name.lower():
            return ToolResult(
                structured_content={
                    "dataset_id": ds["dataset"].id,
                    "dataset_type": ds["type"],
                },
            )
    raise ToolError(
        f"Dataset with name containing '{dataset_name}' not found in project {project_id}."
    )
