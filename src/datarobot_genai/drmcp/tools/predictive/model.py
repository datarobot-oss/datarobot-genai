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
from typing import Annotated
from typing import Any

from datarobot.models.model import Model
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp import dr_mcp_integration_tool
from datarobot_genai.drmcp.tools.clients.datarobot import DataRobotClient
from datarobot_genai.drmcp.tools.clients.datarobot import get_datarobot_access_token

logger = logging.getLogger(__name__)


def model_to_dict(model: Any) -> dict[str, Any]:
    """Convert a DataRobot Model object to a dictionary."""
    try:
        return {
            "id": model.id,
            "model_type": model.model_type,
            "metrics": model.metrics,
        }
    except AttributeError as e:
        logger.warning(f"Failed to access some model attributes: {e}")
        # Return minimal information if some attributes are not accessible
        return {
            "id": getattr(model, "id", "unknown"),
            "model_type": getattr(model, "model_type", "unknown"),
        }


class ModelEncoder(json.JSONEncoder):
    """Custom JSON encoder for DataRobot Model objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Model):
            return model_to_dict(obj)
        return super().default(obj)


@dr_mcp_integration_tool(tags={"predictive", "model", "read", "management", "info"})
async def get_best_model(
    *,
    project_id: Annotated[str, "The DataRobot project ID"] | None = None,
    metric: Annotated[str, "The metric to use for best model selection (e.g., 'AUC', 'LogLoss')"]
    | None = None,
) -> ToolError | ToolResult:
    """Get the best model for a DataRobot project, optionally by a specific metric."""
    if not project_id:
        raise ToolError("Project ID must be provided")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    project = client.Project.get(project_id)
    if not project:
        raise ToolError(f"Project with ID {project_id} not found.")

    leaderboard = project.get_models()
    if not leaderboard:
        raise ToolError("No models found for this project.")

    if metric:
        reverse_sort = metric.upper() in [
            "AUC",
            "ACCURACY",
            "F1",
            "PRECISION",
            "RECALL",
        ]
        leaderboard = sorted(
            leaderboard,
            key=lambda m: m.metrics.get(metric, {}).get(
                "validation", float("-inf") if reverse_sort else float("inf")
            ),
            reverse=reverse_sort,
        )
        logger.info(f"Sorted models by metric: {metric}")

    best_model = leaderboard[0]
    logger.info(f"Found best model {best_model.id} for project {project_id}")

    metric_value = None

    if metric and best_model.metrics and metric in best_model.metrics:
        metric_value = best_model.metrics[metric].get("validation")

    # Include full metrics in the response
    best_model_dict = model_to_dict(best_model)
    best_model_dict["metric"] = metric
    best_model_dict["metric_value"] = metric_value

    return ToolResult(
        structured_content={
            "project_id": project_id,
            "best_model": best_model_dict,
        },
    )


@dr_mcp_integration_tool(tags={"predictive", "model", "read", "scoring", "dataset"})
async def score_dataset_with_model(
    *,
    project_id: Annotated[str, "The DataRobot project ID"] | None = None,
    model_id: Annotated[str, "The DataRobot model ID"] | None = None,
    dataset_url: Annotated[str, "The dataset URL"] | None = None,
) -> ToolError | ToolResult:
    """Score a dataset using a specific DataRobot model."""
    if not project_id:
        raise ToolError("Project ID must be provided")
    if not model_id:
        raise ToolError("Model ID must be provided")
    if not dataset_url:
        raise ToolError("Dataset URL must be provided")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    project = client.Project.get(project_id)
    model = client.Model.get(project, model_id)
    job = model.score(dataset_url)

    return ToolResult(
        structured_content={
            "scoring_job_id": job.id,
            "project_id": project_id,
            "model_id": model_id,
            "dataset_url": dataset_url,
        },
    )


@dr_mcp_integration_tool(tags={"predictive", "model", "read", "management", "list"})
async def list_models(
    *,
    project_id: Annotated[str, "The DataRobot project ID"] | None = None,
) -> ToolError | ToolResult:
    """List all models in a project."""
    if not project_id:
        raise ToolError("Project ID must be provided")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    project = client.Project.get(project_id)
    models = project.get_models()

    return ToolResult(
        structured_content={
            "project_id": project_id,
            "models": [model_to_dict(model) for model in models],
        },
    )
