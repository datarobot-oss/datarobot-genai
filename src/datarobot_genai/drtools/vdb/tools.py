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

"""DataRobot Vector Database (VDB) tools."""

import logging
from typing import Annotated
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp import dr_mcp_integration_tool
from datarobot_genai.drtools.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.clients.datarobot import get_datarobot_access_token

logger = logging.getLogger(__name__)


@dr_mcp_integration_tool(tags={"vdb", "read", "list", "daria"})
async def list_vector_databases() -> ToolResult:
    """List all deployed Vector Databases (VDBs) in DataRobot."""
    token = await get_datarobot_access_token()
    dr_module = DataRobotClient(token).get_client()
    rest_client = dr_module.client.get_client()

    response = rest_client.get(
        "deployments/",
        params={"limit": 100, "modelTargetType": "VectorDatabase"},
    )
    data = response.json()
    vdbs = data.get("data", [])
    next_page = data.get("next")

    return ToolResult(
        structured_content={
            "vector_databases": [
                {
                    "deployment_id": d["id"],
                    "label": d.get("label", ""),
                    "status": d.get("status", ""),
                }
                for d in vdbs
            ],
            "count": len(vdbs),
            "has_more": next_page is not None,
        },
    )


@dr_mcp_integration_tool(tags={"vdb", "read", "query", "search", "daria"})
async def query_vector_database(
    *,
    deployment_id: Annotated[str, "The deployment ID of the Vector Database"] | None = None,
    query: Annotated[str, "The search query"] | None = None,
    num_results: Annotated[int, "Number of results to return"] = 5,
    retrieval_mode: Annotated[
        str, "Retrieval mode: 'similarity' or 'maximal_marginal_relevance'"
    ] = "similarity",
) -> ToolError | ToolResult:
    """Query a DataRobot Vector Database with semantic search."""
    if not deployment_id:
        raise ToolError("Deployment ID must be provided")
    if not query:
        raise ToolError("Query must be provided")

    token = await get_datarobot_access_token()
    dr_module = DataRobotClient(token).get_client()
    rest_client = dr_module.client.get_client()

    payload: dict[str, Any] = {
        "query": query,
        "num_results": num_results,
        "retrieval_mode": retrieval_mode,
    }
    response = rest_client.post(
        f"deployments/{deployment_id}/predictions/",
        json=payload,
    )
    data = response.json()
    documents = data if isinstance(data, list) else data.get("data", [])

    return ToolResult(
        structured_content={
            "deployment_id": deployment_id,
            "documents": documents,
            "count": len(documents),
        },
    )
