"""MCP resource handlers for DataRobot registered models.

Registers:
  model://            — list all registered models
  model://{model_id}  — model details and metrics
"""
from __future__ import annotations

import logging
from typing import Any

from datarobot_genai.drmcp.core.mcp_instance import mcp

logger = logging.getLogger(__name__)


@mcp.resource("model://")
async def list_registered_models() -> list[dict[str, Any]]:
    """List all DataRobot registered models accessible to the current user.

    Returns a list of dicts with id, name, target, and uri fields.
    """
    import datarobot as dr

    client = dr.Client()
    response = client.get("registeredModels/", params={"limit": 100})
    items = response.json().get("data", [])
    return [
        {
            "id": item["id"],
            "name": item.get("name", ""),
            "target": item.get("target"),
            "uri": f"model://{item['id']}",
        }
        for item in items
    ]


@mcp.resource("model://{model_id}")
async def get_registered_model(model_id: str) -> dict[str, Any]:
    """Retrieve registered model details: versions, target, and latest metrics.

    This is a standard MCP resource — any MCP client can read it.
    """
    import datarobot as dr

    client = dr.Client()
    response = client.get(f"registeredModels/{model_id}/")
    model_data = response.json()

    result: dict[str, Any] = {
        "id": model_id,
        "name": model_data.get("name", ""),
        "target": model_data.get("target"),
        "created_at": model_data.get("createdAt"),
        "versions": [],
    }

    try:
        versions_response = client.get(
            f"registeredModels/{model_id}/versions/", params={"limit": 10}
        )
        versions = versions_response.json().get("data", [])
        result["versions"] = [
            {
                "id": v["id"],
                "version_number": v.get("modelVersionNumber"),
                "created_at": v.get("createdAt"),
            }
            for v in versions
        ]
    except Exception as exc:
        logger.debug("Could not fetch versions for model %s: %s", model_id, exc)

    return result
