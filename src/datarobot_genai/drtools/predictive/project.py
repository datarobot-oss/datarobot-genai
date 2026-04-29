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


@tool_metadata(
    tags={"predictive", "project", "read", "management", "list"},
    description=(
        "[Project—discover ids] Use when the user needs their modeling projects as id-to-name "
        "map (no single project_id yet). Read-only. Not for datasets inside one project "
        "(get_project_dataset_by_name), not catalog datasets (list_ai_catalog_items), not "
        "deployments (list_deployments)."
    ),
)
async def list_projects() -> dict[str, Any]:
    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    projects = client.Project.list()
    projects = {p.id: p.project_name for p in projects}

    return projects


@tool_metadata(
    tags={"predictive", "project", "read", "data", "info"},
    description=(
        "[Project—resolve dataset by name] Use when the user names or describes a dataset "
        "already attached to a modeling project (e.g. 'get the holdout dataset', 'dataset named "
        "X') and you need its dataset_id: pass project_id plus dataset_name as a case-insensitive "
        "substring of the dataset display name. Read-only. Returns dataset_id and whether it is "
        "the project source or a prediction upload. Not for listing all projects "
        "(list_projects) or arbitrary catalog lookup."
    ),
)
async def get_project_dataset_by_name(
    *,
    project_id: Annotated[str, "DataRobot modeling project id."],
    dataset_name: Annotated[str, "Substring to match against dataset display names."],
) -> dict[str, Any]:
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
