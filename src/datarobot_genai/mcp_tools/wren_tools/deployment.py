"""DataRobot deployment tools.

Ported from wren-mcp bi_deployment.py. Returns plain dicts instead of
panel-specific DatasetRef/staging types.
"""
from __future__ import annotations

import json
import logging

from datarobot_genai.mcp_tools._registry import register_tool

logger = logging.getLogger(__name__)


async def get_deployment_info(deployment_id: str) -> str:
    """Get deployment details including input features, target, and time series config.

    Returns a JSON string with features, target, is_time_series, and time_series_config.
    """
    import datarobot as dr

    deployment = dr.Deployment.get(deployment_id)
    model = deployment.model

    info: dict = {
        "deployment_id": deployment_id,
        "label": deployment.label,
        "model_id": model.get("id") if model else None,
        "target": None,
        "is_time_series": False,
        "features": [],
        "time_series_config": None,
    }

    try:
        project_id = model.get("project_id") if model else None
        if project_id:
            project = dr.Project.get(project_id)
            info["target"] = project.target

            try:
                spec = dr.DatetimePartitioningSpecification.get(project_id)
                info["is_time_series"] = True
                info["time_series_config"] = {
                    "datetime_partition_column": getattr(
                        spec, "datetime_partition_column", None
                    ),
                    "forecast_window_start": getattr(spec, "forecast_window_start", None),
                    "forecast_window_end": getattr(spec, "forecast_window_end", None),
                }
            except Exception:
                pass  # Not a time series project

        model_id = model.get("id") if model else None
        if model_id and project_id:
            try:
                dr_model = dr.Model.get(project_id, model_id)
                features = dr_model.get_features_used()
                info["features"] = [
                    {"name": f.name, "type": getattr(f, "feature_type", "unknown")}
                    for f in features
                ]
            except Exception:
                pass
    except Exception as exc:
        info["error"] = str(exc)

    return json.dumps(info, default=str)


async def predict_with_deployment(
    deployment_id: str,
    dataset_id: str,
    max_explanations: int = 0,
    threshold_high: float | None = None,
    threshold_low: float | None = None,
) -> dict:
    """Make predictions using a DataRobot deployment on a dataset.

    Returns a dict with prediction rows, columns, and metadata.
    Set max_explanations > 0 to include prediction explanations.
    """
    import datarobot as dr

    deployment = dr.Deployment.get(deployment_id)
    dataset = dr.Dataset.get(dataset_id)
    df = dataset.get_as_dataframe()

    predict_job = deployment.predict(
        df,
        max_explanations=max_explanations if max_explanations > 0 else None,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
    )

    predictions_df = predict_job.get_result_when_complete()
    return {
        "deployment_id": deployment_id,
        "dataset_id": dataset_id,
        "row_count": len(predictions_df),
        "columns": list(predictions_df.columns),
        "predictions": predictions_df.to_dict(orient="records"),
    }


async def deploy_model(
    model_id: str,
    project_id: str,
    label: str,
    description: str = "",
    default_prediction_server_id: str | None = None,
) -> str:
    """Deploy a DataRobot model by creating a deployment.

    Returns a JSON string with deployment_id, label, and model_id.
    """
    import datarobot as dr

    model = dr.Model.get(project_id, model_id)

    registered_model_version = dr.RegisteredModelVersion.create_for_leaderboard_item(
        model_id=model.id,
        registered_model_name=label,
    )

    deployment_kwargs: dict = {
        "registered_model_version_id": registered_model_version.id,
        "label": label,
        "description": description,
    }
    if default_prediction_server_id:
        deployment_kwargs["default_prediction_server_id"] = default_prediction_server_id

    deployment = dr.Deployment.create_from_registered_model_version(**deployment_kwargs)

    return json.dumps(
        {
            "deployment_id": deployment.id,
            "label": deployment.label,
            "model_id": model_id,
        }
    )


async def get_prediction_history(
    deployment_id: str,
    limit: int = 100,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict:
    """Retrieve recent prediction results from a DataRobot deployment.

    Returns rows of historical predictions with optional time range filtering.
    """
    import datarobot as dr

    client = dr.Client()
    params: dict = {"limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    response = client.get(f"deployments/{deployment_id}/predictionResults/", params=params)
    data = response.json()
    rows = data.get("data", [])
    return {
        "deployment_id": deployment_id,
        "row_count": len(rows),
        "rows": rows,
    }


register_tool(
    "get_deployment_info",
    get_deployment_info,
    "Get deployment details including features, target, and time series config.",
    "wren_tools",
)
register_tool(
    "predict_with_deployment",
    predict_with_deployment,
    "Make predictions using a DataRobot deployment on a dataset.",
    "wren_tools",
)
register_tool(
    "deploy_model",
    deploy_model,
    "Deploy a DataRobot model by creating a deployment.",
    "wren_tools",
)
register_tool(
    "get_prediction_history",
    get_prediction_history,
    "Retrieve recent prediction results from a DataRobot deployment.",
    "wren_tools",
)
