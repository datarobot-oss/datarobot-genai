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
from dataclasses import asdict
from dataclasses import dataclass
from typing import Annotated
from typing import Any

import polars as pl
from datarobot.models.model import Model

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.exceptions import ToolError

logger = logging.getLogger(__name__)

# Max target null rate (30%) for time-series eligibility check
_MAX_NULL_RATE_FOR_ELIGIBILITY = 0.3


@dataclass
class ModelInsights:
    """DTO for model detail information returned by get_model_details."""

    model_id: str
    project_id: str
    model_type: str
    featurelist_name: str | None
    target: str
    metric: str
    metrics: dict[str, Any]
    sample_pct: float | None
    feature_impact: list[dict[str, Any]] | None = None
    feature_impact_error: str | None = None
    roc_curve: dict[str, Any] | None = None
    roc_curve_error: str | None = None


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


@tool_metadata(
    tags={"predictive", "model", "read", "management", "info", "daria"},
    description=(
        "[Project—pick best model] Use when the user wants the top leaderboard model for one "
        "modeling project, optionally ranked by a validation metric (e.g. AUC, LogLoss). "
        "Read-only; returns project_id plus best model id, type, and metrics. Not for listing "
        "every model (list_models), not deployment scoring metadata (get_deployment_info), "
        "not full diagnostics (get_model_details)."
    ),
)
async def get_best_model(
    *,
    project_id: Annotated[str, "DataRobot modeling project id."],
    metric: Annotated[
        str, "Optional leaderboard sort key (e.g. AUC, LogLoss); omit for default order."
    ]
    | None = None,
) -> dict[str, Any]:
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

    return {
        "project_id": project_id,
        "best_model": best_model_dict,
    }


@tool_metadata(
    tags={"predictive", "model", "read", "scoring", "dataset"},
    description=(
        "[Project—model vs catalog] Use when the user wants to score an AI Catalog dataset with a "
        "specific leaderboard model inside a modeling project (they give project_id, model_id, and "
        "catalog dataset_id). This is not deployment batch scoring (see predict_by_ai_catalog) and "
        "not inline CSV in chat (see predict_realtime). Starts an async project scoring job; "
        "returns scoring_job_id and related dataset ids, not prediction rows."
    ),
)
async def score_dataset_with_model(
    *,
    project_id: Annotated[str, "Modeling project that owns the model."],
    model_id: Annotated[str, "Leaderboard model id from list_models."],
    dataset_id: Annotated[str, "AI Catalog dataset id to score (tabular)."],
) -> dict[str, Any]:
    if not project_id:
        raise ToolError("Project ID must be provided")
    if not model_id:
        raise ToolError("Model ID must be provided")
    if not dataset_id or not dataset_id.strip():
        raise ToolError("Dataset ID must be provided")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    project = client.Project.get(project_id)
    dr_model = client.Model.get(project, model_id)
    catalog_dataset = client.Dataset.get(dataset_id)
    prediction_dataset = project.upload_dataset_from_catalog(dataset_id=catalog_dataset.id)
    job = dr_model.request_predictions(dataset_id=prediction_dataset.id)

    return {
        "message": "Scoring job started",
        "scoring_job_id": job.id,
        "catalog_dataset_id": catalog_dataset.id,
        "prediction_dataset_id": prediction_dataset.id,
        "project_id": project_id,
        "model_id": model_id,
    }


@tool_metadata(
    tags={"predictive", "model", "read", "management", "list", "daria"},
    description=(
        "[Project—list models] Use when the user needs every trained leaderboard model for a "
        "project (ids, types, metrics). Read-only. Not the same as picking only the best model "
        "(get_best_model). Follow with get_model_details, deploy_model, or "
        "score_dataset_with_model using a chosen model_id."
    ),
)
async def list_models(
    *,
    project_id: Annotated[str, "DataRobot modeling project id."],
) -> dict[str, Any]:
    if not project_id:
        raise ToolError("Project ID must be provided")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    project = client.Project.get(project_id)
    models = project.get_models()

    return {
        "project_id": project_id,
        "models": [model_to_dict(model) for model in models],
    }


@tool_metadata(
    tags={"predictive", "model", "read", "details", "info", "daria"},
    description=(
        "[Project—model diagnostics] Use when the user asks for training-time detail on one "
        "leaderboard model: target, project metric, validation metrics, optional feature impact "
        "and ROC. Read-only. For MLOps deployment input columns and prediction contract, use "
        "get_deployment_info instead. For ROC-only or lift-only charts with explicit source "
        "fold, see get_model_roc_curve / get_model_lift_chart."
    ),
)
async def get_model_details(
    *,
    project_id: Annotated[str, "DataRobot modeling project id."],
    model_id: Annotated[str, "Leaderboard model id."],
    include_feature_impact: Annotated[
        bool, "If true, request or return per-feature impact."
    ] = True,
    include_roc_curve: Annotated[
        bool, "If true, include validation ROC points (classification)."
    ] = False,
) -> dict[str, Any]:
    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    project = client.Project.get(project_id)
    model = client.Model.get(project=project, model_id=model_id)

    insights = ModelInsights(
        model_id=model_id,
        project_id=project_id,
        model_type=model.model_type,
        featurelist_name=getattr(model, "featurelist_name", None),
        target=project.target,
        metric=project.metric,
        metrics=model.metrics,
        sample_pct=getattr(model, "sample_pct", None),
    )

    if include_feature_impact:
        try:
            insights.feature_impact = model.get_or_request_feature_impact()
        except Exception as exc:
            insights.feature_impact_error = str(exc)

    if include_roc_curve:
        try:
            roc = model.get_roc_curve(source="validation")
            insights.roc_curve = {
                "source": "validation",
                "roc_points": roc.roc_points if hasattr(roc, "roc_points") else [],
            }
        except Exception as exc:
            insights.roc_curve_error = str(exc)

    return {k: v for k, v in asdict(insights).items() if v is not None}


@tool_metadata(
    tags={"predictive", "model", "read", "timeseries", "validation", "daria"},
    description=(
        "[Catalog—time series readiness] Use before starting time-series Autopilot: checks an "
        "AI Catalog dataset for row count, parsable datetime column, target null rate, optional "
        "multiseries id column. Read-only; returns ELIGIBLE or NOT_ELIGIBLE with reasons; does "
        "not train. Not general EDA (get_exploratory_insights / analyze_dataset) and not "
        "tabular-only Autopilot start (start_autopilot without TS-specific checks)."
    ),
)
async def is_eligible_for_timeseries_training(
    *,
    dataset_id: Annotated[str, "AI Catalog dataset id."],
    datetime_column: Annotated[str, "Column name with timestamps."],
    target_column: Annotated[str, "Column name to forecast."],
    series_id_column: Annotated[
        str, "Multiseries: column distinguishing each series; omit for single series."
    ]
    | None = None,
) -> dict[str, Any]:
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
        pandas_df = dataset.get_as_dataframe()
        df = pl.from_pandas(pandas_df)
    except Exception as exc:
        raise ToolError(f"Could not load dataset: {exc}")

    row_count = df.height
    infos.append(f"Row count: {row_count}")
    if row_count < 100:
        errors.append(f"Too few rows ({row_count}): time series training requires at least 100.")

    if datetime_column not in df.columns:
        errors.append(f"Datetime column '{datetime_column}' not found in dataset.")
    else:
        try:
            col = pl.col(datetime_column)
            dtype = df[datetime_column].dtype
            if dtype == pl.Utf8:
                df = df.with_columns(col.str.to_datetime())
            else:
                df = df.with_columns(col.cast(pl.Datetime))
            infos.append(f"Datetime column '{datetime_column}' parsed successfully.")
        except Exception as exc:
            errors.append(f"Datetime column '{datetime_column}' could not be parsed: {exc}")

    if series_id_column and series_id_column not in df.columns:
        errors.append(f"Series ID column '{series_id_column}' not found in dataset.")

    if target_column not in df.columns:
        errors.append(f"Target column '{target_column}' not found in dataset.")

    null_pct = df[target_column].null_count() / row_count if target_column in df.columns else None
    if null_pct is not None:
        infos.append(f"Target null rate: {null_pct:.1%}")
        if null_pct > _MAX_NULL_RATE_FOR_ELIGIBILITY:
            errors.append(f"Target column has {null_pct:.1%} null values (>30%).")

    status = "ELIGIBLE" if not errors else "NOT_ELIGIBLE"

    return {
        "status": status,
        "errors": errors,
        "info": infos,
    }
