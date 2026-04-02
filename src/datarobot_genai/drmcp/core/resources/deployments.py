"""MCP resource handlers for DataRobot deployments.

Registers:
  deployment://            — list all deployments
  deployment://{deployment_id} — deployment info with model, target, features
"""
from __future__ import annotations

import logging
from typing import Any

from datarobot_genai.drmcp.core.mcp_instance import mcp

logger = logging.getLogger(__name__)


@mcp.resource("deployment://")
async def list_deployments() -> list[dict[str, Any]]:
    """List all DataRobot deployments accessible to the current user.

    Returns a list of dicts with id, label, status, and uri fields.
    """
    import datarobot as dr

    deployments = dr.Deployment.list()
    return [
        {
            "id": d.id,
            "label": d.label,
            "status": getattr(d, "status", None),
            "uri": f"deployment://{d.id}",
        }
        for d in deployments
    ]


@mcp.resource("deployment://{deployment_id}")
async def get_deployment(deployment_id: str) -> dict[str, Any]:
    """Retrieve deployment details: model info, target, features, and time series config.

    This is a standard MCP resource — any MCP client can read it.
    """
    import datarobot as dr

    deployment = dr.Deployment.get(deployment_id)
    model = deployment.model or {}

    result: dict[str, Any] = {
        "id": deployment_id,
        "label": deployment.label,
        "status": getattr(deployment, "status", None),
        "model_id": model.get("id"),
        "model_type": model.get("type"),
        "target": None,
        "is_time_series": False,
        "features": [],
        "time_series_config": None,
    }

    project_id = model.get("project_id")
    if project_id:
        try:
            project = dr.Project.get(project_id)
            result["target"] = project.target
        except Exception as exc:
            logger.debug("Could not fetch project %s: %s", project_id, exc)

        try:
            spec = dr.DatetimePartitioningSpecification.get(project_id)
            result["is_time_series"] = True
            result["time_series_config"] = {
                "datetime_partition_column": getattr(spec, "datetime_partition_column", None),
                "forecast_window_start": getattr(spec, "forecast_window_start", None),
                "forecast_window_end": getattr(spec, "forecast_window_end", None),
            }
        except Exception:
            pass  # Not a time series project

        model_id = model.get("id")
        if model_id:
            try:
                dr_model = dr.Model.get(project_id, model_id)
                features = dr_model.get_features_used()
                result["features"] = [
                    {"name": f.name, "type": getattr(f, "feature_type", "unknown")}
                    for f in features
                ]
            except Exception as exc:
                logger.debug("Could not fetch features for model %s: %s", model_id, exc)

    return result
