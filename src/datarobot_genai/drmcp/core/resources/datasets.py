"""MCP resource handlers for DataRobot datasets.

Registers:
  dataset://            — list all accessible datasets
  dataset://{dataset_id} — metadata + sample rows for a dataset
"""
from __future__ import annotations

import logging
from typing import Any

from datarobot_genai.drmcp.core.mcp_instance import mcp

logger = logging.getLogger(__name__)


@mcp.resource("dataset://")
async def list_datasets() -> list[dict[str, Any]]:
    """List all DataRobot AI Catalog datasets accessible to the current user.

    Returns a list of dicts with id, name, and uri fields.
    """
    import datarobot as dr

    client = dr.Client()
    response = client.get("catalogItems/", params={"limit": 100})
    items = response.json().get("data", [])
    return [
        {
            "id": item["id"],
            "name": item.get("name", ""),
            "uri": f"dataset://{item['id']}",
        }
        for item in items
    ]


@mcp.resource("dataset://{dataset_id}")
async def get_dataset(dataset_id: str) -> dict[str, Any]:
    """Retrieve dataset metadata: schema, row count, columns, and sample rows.

    This is a standard MCP resource — any MCP client can read it.
    """
    import datarobot as dr

    dataset = dr.Dataset.get(dataset_id)
    result: dict[str, Any] = {
        "id": dataset.id,
        "name": dataset.name,
        "created_at": str(dataset.created_at),
        "row_count": getattr(dataset, "row_count", None),
    }

    try:
        df = dataset.get_as_dataframe()
        result["columns"] = [
            {"name": col, "type": str(df[col].dtype)} for col in df.columns
        ]
        result["sample_rows"] = df.head(5).to_dict(orient="records")
    except Exception as exc:
        logger.warning("Could not load dataframe for dataset %s: %s", dataset_id, exc)
        result["columns"] = []
        result["sample_rows"] = []
        result["sample_error"] = str(exc)

    return result
