"""DataRobot ML model and AutoPilot tools.

Ported from wren-mcp bi_model.py and bi_forecasting.py. Returns plain
dicts/strings instead of panel-specific types.
"""
from __future__ import annotations

import json
import logging

from datarobot_genai.mcp_tools._registry import register_tool

logger = logging.getLogger(__name__)


async def list_models(project_id: str) -> str:
    """List all models in a DataRobot project with metrics and feature info.

    Returns a formatted string summary of models on the leaderboard.
    """
    import datarobot as dr

    project = dr.Project.get(project_id)
    models = project.get_models()

    lines = [f"Models for project '{project.project_name}' ({project_id}):"]
    for m in models:
        metric_val = None
        if project.metric and m.metrics:
            metric_data = m.metrics.get(project.metric, {})
            metric_val = metric_data.get("validation") or metric_data.get("crossValidation")
        lines.append(
            f"  [{m.id}] {m.model_type}"
            f" | {project.metric}={metric_val}"
            f" | features={m.featurelist_name}"
        )
    return "\n".join(lines)


async def get_model_info(
    project_id: str,
    model_id: str,
    include_feature_impact: bool = True,
    include_roc_curve: bool = False,
) -> str:
    """Get detailed information about a DataRobot model.

    Optionally includes feature impact and ROC curve data.
    Returns a JSON string with comprehensive model details.
    """
    import datarobot as dr

    model = dr.Model.get(project_id, model_id)
    project = dr.Project.get(project_id)

    info: dict = {
        "model_id": model_id,
        "project_id": project_id,
        "model_type": model.model_type,
        "featurelist_name": model.featurelist_name,
        "target": project.target,
        "metric": project.metric,
        "metrics": model.metrics,
        "sample_pct": model.sample_pct,
    }

    if include_feature_impact:
        try:
            fi_job = model.request_feature_impact()
            fi = fi_job.get_result_when_complete()
            info["feature_impact"] = [
                {"feature": f.feature_name, "impact": f.impact_normalized} for f in fi
            ]
        except Exception as exc:
            info["feature_impact_error"] = str(exc)

    if include_roc_curve:
        try:
            roc = dr.RocCurve.get(project_id, model_id)
            info["roc_curve"] = {
                "source": roc.source,
                "auc": getattr(roc, "auc", None),
            }
        except Exception as exc:
            info["roc_curve_error"] = str(exc)

    return json.dumps(info, default=str)


async def run_autopilot(
    dataset_id: str,
    target: str,
    project_name: str | None = None,
    mode: str = "quick",
    metric: str | None = None,
    worker_count: int = -1,
) -> str:
    """Train ML models using DataRobot AutoPilot on a dataset.

    mode options: 'quick', 'comprehensive', 'manual'.
    Returns the project URL when training starts.
    """
    import datarobot as dr

    dataset = dr.Dataset.get(dataset_id)
    name = project_name or f"Autopilot: {dataset.name} → {target}"
    project = dr.Project.create_from_dataset(
        dataset_id=dataset_id,
        project_name=name,
    )
    project.set_target(
        target=target,
        metric=metric,
        worker_count=worker_count,
        mode=getattr(dr.enums.AUTOPILOT_MODE, mode.upper(), dr.enums.AUTOPILOT_MODE.QUICK),
    )
    return f"Autopilot started: {project.id} — {dr.Client().endpoint}/projects/{project.id}"


async def is_eligible_for_timeseries_training(
    dataset_id: str,
    datetime_column: str,
    target_column: str,
    series_id_column: str | None = None,
) -> str:
    """Check if a dataset is eligible for DataRobot time series training.

    Validates row count, datetime column, series ID, duplicates, and frequency.
    Returns a validation report string with errors and informational messages.
    """
    import datarobot as dr

    dataset = dr.Dataset.get(dataset_id)
    errors: list[str] = []
    infos: list[str] = []

    try:
        df = dataset.get_as_dataframe()
    except Exception as exc:
        return f"ERROR: Could not load dataset: {exc}"

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
        if null_pct > 0.3:
            errors.append(f"Target column has {null_pct:.1%} null values (>30%).")

    status = "ELIGIBLE" if not errors else "NOT ELIGIBLE"
    lines = [f"Time series eligibility: {status}", ""]
    if errors:
        lines.append("Errors:")
        lines.extend(f"  ✗ {e}" for e in errors)
    if infos:
        lines.append("Info:")
        lines.extend(f"  ℹ {i}" for i in infos)
    return "\n".join(lines)


register_tool(
    "list_models",
    list_models,
    "List all models in a DataRobot project with metrics and feature info.",
    "wren_tools",
)
register_tool(
    "get_model_info",
    get_model_info,
    "Get detailed information about a DataRobot model.",
    "wren_tools",
)
register_tool(
    "run_autopilot",
    run_autopilot,
    "Train ML models using DataRobot AutoPilot on a dataset.",
    "wren_tools",
)
register_tool(
    "is_eligible_for_timeseries_training",
    is_eligible_for_timeseries_training,
    "Check if a dataset is eligible for DataRobot time series training.",
    "wren_tools",
)
