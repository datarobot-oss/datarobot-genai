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
from datarobot_genai.drtools.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.clients.datarobot import get_datarobot_access_token

logger = logging.getLogger(__name__)

# Max target null rate (30%) for time-series eligibility check
_MAX_NULL_RATE_FOR_ELIGIBILITY = 0.3


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


@dr_mcp_integration_tool(tags={"predictive", "model", "read", "management", "info", "daria"})
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


@dr_mcp_integration_tool(tags={"predictive", "model", "read", "management", "list", "daria"})
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


@dr_mcp_integration_tool(tags={"predictive", "model", "read", "details", "info", "daria"})
async def get_model_details(
    *,
    project_id: Annotated[str, "The DataRobot project ID"] | None = None,
    model_id: Annotated[str, "The DataRobot model ID"] | None = None,
    include_feature_impact: Annotated[bool, "Whether to include feature impact data"] = True,
    include_roc_curve: Annotated[bool, "Whether to include ROC curve data"] = False,
) -> ToolError | ToolResult:
    """Get detailed information about a DataRobot model, optionally with feature impact and ROC."""
    if not project_id:
        raise ToolError("Project ID must be provided")
    if not model_id:
        raise ToolError("Model ID must be provided")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    project = client.Project.get(project_id)
    model = client.Model.get(project=project, model_id=model_id)

    info: dict[str, Any] = {
        "model_id": model_id,
        "project_id": project_id,
        "model_type": model.model_type,
        "featurelist_name": getattr(model, "featurelist_name", None),
        "target": project.target,
        "metric": project.metric,
        "metrics": model.metrics,
        "sample_pct": getattr(model, "sample_pct", None),
    }

    if include_feature_impact:
        try:
            model.request_feature_impact()
            fi = model.get_or_request_feature_impact()
            info["feature_impact"] = fi
        except Exception as exc:
            info["feature_impact_error"] = str(exc)

    if include_roc_curve:
        try:
            roc = model.get_roc_curve(source="validation")
            info["roc_curve"] = {
                "source": "validation",
                "roc_points": roc.roc_points if hasattr(roc, "roc_points") else [],
            }
        except Exception as exc:
            info["roc_curve_error"] = str(exc)

    return ToolResult(structured_content=info)


@dr_mcp_integration_tool(tags={"predictive", "model", "read", "timeseries", "validation", "daria"})
async def is_eligible_for_timeseries_training(
    *,
    dataset_id: Annotated[str, "The ID of the DataRobot dataset to validate"] | None = None,
    datetime_column: Annotated[str, "The name of the datetime column"] | None = None,
    target_column: Annotated[str, "The name of the target column"] | None = None,
    series_id_column: Annotated[str, "The name of the series ID column"] | None = None,
) -> ToolError | ToolResult:
    """Check if a dataset is eligible for DataRobot time series training."""
    if not dataset_id:
        raise ToolError("Dataset ID must be provided")
    if not datetime_column:
        raise ToolError("Datetime column must be provided")
    if not target_column:
        raise ToolError("Target column must be provided")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    dataset = client.Dataset.get(dataset_id)

    errors: list[str] = []
    infos: list[str] = []

    try:
        df = dataset.get_as_dataframe()
    except Exception as exc:
        raise ToolError(f"Could not load dataset: {exc}")

    row_count = len(df)
    infos.append(f"Row count: {row_count}")
    if row_count < 100:
        errors.append(f"Too few rows ({row_count}): time series training requires at least 100.")

    if datetime_column not in df.columns:
        errors.append(f"Datetime column '{datetime_column}' not found in dataset.")
    else:
        try:
            import pandas as pd

            df[datetime_column] = pd.to_datetime(df[datetime_column])
            infos.append(f"Datetime column '{datetime_column}' parsed successfully.")
        except Exception as exc:
            errors.append(f"Datetime column '{datetime_column}' could not be parsed: {exc}")

    if series_id_column and series_id_column not in df.columns:
        errors.append(f"Series ID column '{series_id_column}' not found in dataset.")

    if target_column not in df.columns:
        errors.append(f"Target column '{target_column}' not found in dataset.")

    null_pct = df[target_column].isnull().mean() if target_column in df.columns else None
    if null_pct is not None:
        infos.append(f"Target null rate: {null_pct:.1%}")
        if null_pct > _MAX_NULL_RATE_FOR_ELIGIBILITY:
            errors.append(f"Target column has {null_pct:.1%} null values (>30%).")

    status = "ELIGIBLE" if not errors else "NOT_ELIGIBLE"

    return ToolResult(
        structured_content={
            "status": status,
            "errors": errors,
            "info": infos,
        },
    )
