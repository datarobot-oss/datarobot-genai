"""DataRobot Vector Database (VDB) tools.

Ported from wren-mcp bi_vdb.py.
"""
from __future__ import annotations

import logging
from typing import Any

from datarobot_genai.mcp_tools._registry import register_tool

logger = logging.getLogger(__name__)


async def list_vector_databases() -> list[dict[str, Any]]:
    """List all deployed Vector Databases (VDBs) in DataRobot.

    Returns a list of dicts with deployment_id, label, and status.
    """
    import datarobot as dr

    client = dr.Client()
    response = client.get("deployments/", params={"limit": 100})
    all_deployments = response.json().get("data", [])

    vdbs = [
        d
        for d in all_deployments
        if d.get("capabilities", {}).get("supportsVectorDatabaseQuerying")
        or d.get("model", {}).get("targetType") == "VectorDatabase"
    ]
    return [
        {
            "deployment_id": d["id"],
            "label": d.get("label", ""),
            "status": d.get("status", ""),
        }
        for d in vdbs
    ]


async def query_vector_database(
    deployment_id: str,
    query: str,
    num_results: int = 5,
    retrieval_mode: str = "similarity",
) -> list[dict[str, Any]]:
    """Query a DataRobot Vector Database with semantic search.

    retrieval_mode options: 'similarity', 'maximal_marginal_relevance'.
    Returns a list of documents with content and metadata.
    """
    import datarobot as dr
    from datarobot_predict.deployment import DataRobotPredictionClient

    deployment = dr.Deployment.get(deployment_id)
    client = DataRobotPredictionClient(deployment=deployment)

    payload = {
        "query": query,
        "num_results": num_results,
        "retrieval_mode": retrieval_mode,
    }
    response = client.predict(payload)
    return response if isinstance(response, list) else response.get("data", [])


register_tool(
    "list_vector_databases",
    list_vector_databases,
    "List all deployed Vector Databases (VDBs) in DataRobot.",
    "wren_tools",
)
register_tool(
    "query_vector_database",
    query_vector_database,
    "Query a DataRobot Vector Database with semantic search.",
    "wren_tools",
)
