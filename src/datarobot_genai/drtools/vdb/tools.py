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

import datarobot as dr
from datarobot.errors import ClientError

from datarobot_genai.drmcputils.client_exceptions import raise_tool_error_for_client_error
from datarobot_genai.drmcputils.clients.datarobot import ThreadSafeDataRobotClient
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.pagination import PAGINATION_MAX
from datarobot_genai.drtools.pagination import clamp_limit
from datarobot_genai.drtools.pagination import merge_pagination_metadata

logger = logging.getLogger(__name__)


@tool_metadata(
    tags={"vdb", "read", "list", "daria"},
    description=(
        "[VDB—discover deployments] Use when the user needs deployed Vector Databases (VDBs) as "
        "id/label/status records. Read-only. Filters DataRobot deployments to "
        "modelTargetType=VectorDatabase. Not predictive deployments (deployment_get_list), not "
        "AI Catalog datasets (catalog_list_datasets). Next step: vdb_query."
    ),
)
async def vdb_list(
    *,
    offset: Annotated[
        int | None,
        "Skip this many VDBs (0-based). Use with limit for paged listing; omit for all.",
    ] = None,
    limit: Annotated[
        int,
        ("Max VDBs to return (default 100). Values above 100 are rejected; use offset to page."),
    ] = PAGINATION_MAX,
) -> dict[str, Any]:
    if offset is not None and offset < 0:
        raise ToolError("offset must be non-negative", kind=ToolErrorKind.VALIDATION)

    limit, message = clamp_limit(limit)

    params: dict[str, Any] = {"limit": limit, "modelTargetType": "VectorDatabase"}
    if offset is not None:
        params["offset"] = offset

    with ThreadSafeDataRobotClient().request_user_client():
        rest_client = dr.client.get_client()
        try:
            response = rest_client.get("deployments/", params=params)
        except ClientError as e:
            raise_tool_error_for_client_error(e)
        api_response = response.json()
        vdbs = api_response.get("data", [])
        if not isinstance(vdbs, list):
            vdbs = []

    final_results: dict[str, Any] = {
        "vector_databases": [
            {
                "deployment_id": d["id"],
                "label": d.get("label", ""),
                "status": d.get("status", ""),
            }
            for d in vdbs
        ],
        "count": len(vdbs),
    }
    return merge_pagination_metadata(
        final_results=final_results,
        api_response=api_response,
        message=message,
        offset=offset,
        limit=limit,
    )


@tool_metadata(
    tags={"vdb", "read", "query", "search", "daria"},
    description=(
        "[VDB—semantic search] Use when the user wants to retrieve documents from a deployed "
        "Vector Database via semantic similarity (deployment_id from vdb_list). "
        "Returns matched documents with metadata. Read-only. Not deployment metadata "
        "(deployment_get_info), not predictive scoring (predict_*)."
    ),
)
async def vdb_query(
    *,
    deployment_id: Annotated[str, "The deployment ID of the Vector Database"] | None = None,
    query: Annotated[str, "The search query"] | None = None,
    num_results: Annotated[int, "Number of results to return"] = 5,
    retrieval_mode: Annotated[
        str, "Retrieval mode: 'similarity' or 'maximal_marginal_relevance'"
    ] = "similarity",
) -> dict[str, Any]:
    if not deployment_id:
        raise ToolError("Deployment ID must be provided", kind=ToolErrorKind.VALIDATION)
    if not query:
        raise ToolError("Query must be provided", kind=ToolErrorKind.VALIDATION)

    payload: dict[str, Any] = {
        "query": query,
        "num_results": num_results,
        "retrieval_mode": retrieval_mode,
    }

    with ThreadSafeDataRobotClient().request_user_client():
        rest_client = dr.client.get_client()
        try:
            response = rest_client.post(
                f"deployments/{deployment_id}/predictions/",
                json=payload,
            )
        except ClientError as e:
            raise_tool_error_for_client_error(e)
        data = response.json()
        documents = data if isinstance(data, list) else data.get("data", [])

    return {
        "deployment_id": deployment_id,
        "documents": documents,
        "count": len(documents),
    }
