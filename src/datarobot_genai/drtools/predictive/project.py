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
from typing import Any

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.exceptions import ToolError

logger = logging.getLogger(__name__)


@tool_metadata(tags={"predictive", "project", "read", "management", "list"})
async def list_projects(
    *,
    offset: Annotated[
        int | None,
        "Projects to skip (0-based). Use with limit for a page of results.",
    ] = None,
    limit: Annotated[
        int | None,
        "Maximum projects to return. Omit to use the API default (up to 1000 results).",
    ] = None,
) -> dict[str, Any]:
    """List DataRobot projects. Pass offset/limit to page results."""
    if offset is not None and offset < 0:
        raise ToolError("offset must be non-negative")
    if limit is not None and limit < 1:
        raise ToolError("limit must be at least 1 when provided")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    list_kwargs: dict[str, int] = {}
    if offset is not None:
        list_kwargs["offset"] = offset
    if limit is not None:
        list_kwargs["limit"] = limit
    projects = client.Project.list(**list_kwargs) if list_kwargs else client.Project.list()
    projects_dict = {p.id: p.project_name for p in projects}

    if list_kwargs:
        return {
            "projects": projects_dict,
            "offset": list_kwargs.get("offset", 0),
            "limit": list_kwargs.get("limit"),
            "returned_count": len(projects),
        }
    return projects_dict


@tool_metadata(tags={"predictive", "project", "read", "data", "info"})
async def get_project_dataset_by_name(
    *,
    project_id: Annotated[str, "The ID of the DataRobot project."],
    dataset_name: Annotated[str, "The name of the dataset to find (e.g., 'training', 'holdout')."],
) -> dict[str, Any]:
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
            return {
                "dataset_id": ds["dataset"].id,
                "dataset_type": ds["type"],
            }
    raise ToolError(
        f"Dataset with name containing '{dataset_name}' not found in project {project_id}."
    )
