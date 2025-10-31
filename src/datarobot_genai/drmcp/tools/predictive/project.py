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

import json
import logging

from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool

logger = logging.getLogger(__name__)


@dr_mcp_tool(tags={"project", "management", "list"})
async def list_projects() -> str:
    """
    List all DataRobot projects for the authenticated user.

    Returns
    -------
        A string summary of the user's DataRobot projects.
    """
    client = get_sdk_client()
    projects = client.Project.list()
    if not projects:
        return "No projects found."
    return "\n".join(f"{p.id}: {p.project_name}" for p in projects)


@dr_mcp_tool(tags={"project", "data", "info"})
async def get_project_dataset_by_name(project_id: str, dataset_name: str) -> str:
    """
    Get a dataset ID by name for a given project.

    Args:
        project_id: The ID of the DataRobot project.
        dataset_name: The name of the dataset to find (e.g., 'training', 'holdout').

    Returns
    -------
        The dataset ID and the dataset type (source or prediction) as a string, or an error message.
    """
    client = get_sdk_client()
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
            return json.dumps(
                {
                    "dataset_id": ds["dataset"].id,
                    "dataset_type": ds["type"],
                    "ui_panel": ["dataset"],
                },
                indent=2,
            )
    return f"Dataset with name containing '{dataset_name}' not found in project {project_id}."
