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
from typing import Any

from datarobot.models.model import Model

from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool

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


@dr_mcp_tool(tags={"model", "management", "info"})
async def get_best_model(project_id: str, metric: str | None = None) -> str:
    """
    Get the best model for a DataRobot project, optionally by a specific metric.

    Args:
        project_id: The ID of the DataRobot project.
        metric: (Optional) The metric to use for best model selection (e.g., 'AUC', 'LogLoss').

    Returns
    -------
        A formatted string describing the best model.

    Raises
    ------
        Exception: If project not found or no models exist in the project.
    """
    client = get_sdk_client()
    project = client.Project.get(project_id)
    if not project:
        logger.error(f"Project with ID {project_id} not found")
        raise Exception(f"Project with ID {project_id} not found.")

    leaderboard = project.get_models()
    if not leaderboard:
        logger.info(f"No models found for project {project_id}")
        raise Exception("No models found for this project.")

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

    # Format the response as a human-readable string
    metric_info = ""
    if metric and best_model.metrics and metric in best_model.metrics:
        metric_value = best_model.metrics[metric].get("validation")
        if metric_value is not None:
            metric_info = f" with {metric}: {metric_value:.2f}"

    return f"Best model: {best_model.model_type}{metric_info}"


@dr_mcp_tool(tags={"model", "prediction", "scoring"})
async def score_dataset_with_model(project_id: str, model_id: str, dataset_url: str) -> str:
    """
    Score a dataset using a specific DataRobot model.

    Args:
        project_id: The ID of the DataRobot project.
        model_id: The ID of the DataRobot model to use for scoring.
        dataset_url: The URL to the dataset to score (must be accessible to DataRobot).

    Returns
    -------
        A string summary of the scoring job or a meaningful error message.
    """
    client = get_sdk_client()
    project = client.Project.get(project_id)
    model = client.Model.get(project, model_id)
    job = model.score(dataset_url)
    logger.info(f"Started scoring job {job.id} for model {model_id}")
    return f"Scoring job started: {job.id}"


@dr_mcp_tool(tags={"model", "management", "list"})
async def list_models(project_id: str) -> str:
    """
    List all models in a project.

    Args:
        project_id: The ID of the DataRobot project.

    Returns
    -------
        A string summary of the models in the project.
    """
    client = get_sdk_client()
    project = client.Project.get(project_id)
    models = project.get_models()
    return json.dumps(models, indent=2, cls=ModelEncoder)
