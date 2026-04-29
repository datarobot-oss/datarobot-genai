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

import base64
import binascii
import io
import logging
from typing import Annotated
from typing import Any

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.utils import is_valid_url

logger = logging.getLogger(__name__)


def _serialize_datastore_params(params: Any) -> dict[str, Any]:
    """Return JSON-serializable params; DataRobot SDK uses DataStoreParameters, not a dict."""
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    collect = getattr(params, "collect_payload", None)
    if callable(collect):
        payload = collect()
        if isinstance(payload, dict):
            return payload
    return {}


@tool_metadata(
    tags={"predictive", "data", "write", "upload", "catalog", "daria"},
    description=(
        "[Catalog—register new data] Use when the user has file bytes or a public HTTPS URL "
        "and needs a new AI Catalog dataset_id (nothing registered yet). Exactly one of "
        "base64 file content or file_url. Returns dataset id, version id, and name for "
        "Autopilot, scoring, or inspection. Not for listing existing items "
        "(list_ai_catalog_items), not external DB browsing (list_datastores)."
    ),
)
async def upload_dataset_to_ai_catalog(
    *,
    file_content_base64: Annotated[
        str,
        "Base64-encoded file bytes (e.g. CSV). Mutually exclusive with file_url.",
    ]
    | None = None,
    dataset_filename: Annotated[
        str,
        "Filename for base64 upload; include extension (e.g. sales.csv).",
    ] = "data.csv",
    file_url: Annotated[
        str,
        "Public HTTPS URL of the file to register. Mutually exclusive with file_content_base64.",
    ]
    | None = None,
) -> dict[str, Any]:
    if not file_content_base64 and not file_url:
        raise ToolError("Either file_content_base64 or file_url must be provided.")
    if file_content_base64 and file_url:
        raise ToolError("Please provide either file_content_base64 or file_url, not both.")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    catalog_item = None
    if file_content_base64 is not None:
        raw_b64 = file_content_base64.strip()
        if not raw_b64:
            raise ToolError("Argument validation error: 'file_content_base64' cannot be empty.")
        try:
            raw = base64.b64decode(raw_b64)
        except binascii.Error as e:
            raise ToolError("Invalid base64 in file_content_base64.") from e
        if not raw:
            raise ToolError("Decoded file content is empty.")
        fname = (
            dataset_filename.strip()
            if dataset_filename and dataset_filename.strip()
            else "data.csv"
        )
        buffer = io.BytesIO(raw)
        buffer.name = fname
        catalog_item = client.Dataset.create_from_file(filelike=buffer)
    else:
        if file_url is None or not is_valid_url(file_url):
            logger.error("Invalid file URL: %s", file_url)
            raise ToolError(f"Invalid file URL: {file_url}")
        catalog_item = client.Dataset.create_from_url(file_url)

    if not catalog_item:
        raise ToolError("Failed to upload dataset.")

    return {
        "dataset_id": catalog_item.id,
        "dataset_version_id": catalog_item.version_id,
        "dataset_name": catalog_item.name,
    }


@tool_metadata(
    tags={"predictive", "data", "read", "list", "catalog", "daria"},
    description=(
        "[Catalog—list datasets] Use when the user needs catalog datasets they already have "
        "as id-to-name map. Read-only. Not project-attached datasets "
        "(get_project_dataset_by_name), not modeling projects (list_projects). Follow with "
        "get_dataset_details or scoring tools that take dataset_id."
    ),
)
async def list_ai_catalog_items() -> dict[str, Any]:
    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    datasets = client.Dataset.list()

    if not datasets:
        logger.info("No AI Catalog items found")
        return {"datasets": []}

    datasets_dict = {ds.id: ds.name for ds in datasets}

    return {
        "datasets": datasets_dict,
        "count": len(datasets),
    }


@tool_metadata(
    tags={"predictive", "data", "read", "dataset", "metadata", "daria"},
    description=(
        "[Catalog—one dataset metadata] Use when you already have catalog dataset_id and "
        "need name, row count, timestamps, optional column list and sample rows. Read-only. "
        "Lighter than full EDA (analyze_dataset / get_exploratory_insights); not datastore "
        "SQL (query_datastore)."
    ),
)
async def get_dataset_details(
    *,
    dataset_id: Annotated[str, "AI Catalog dataset id (from list_ai_catalog_items or upload)."]
    | None = None,
    include_sample: Annotated[
        bool, "If true, include columns and up to sample_rows example rows."
    ] = True,
    sample_rows: Annotated[int, "Max sample rows when include_sample is true."] = 10,
) -> dict[str, Any]:
    if not dataset_id:
        raise ToolError("Dataset ID must be provided")

    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    dataset = client.Dataset.get(dataset_id)

    result: dict = {
        "id": dataset.id,
        "name": dataset.name,
        "created_at": str(dataset.created_at),
        "row_count": getattr(dataset, "row_count", None),
    }
    if include_sample:
        try:
            df = dataset.get_raw_sample_data()
            result["columns"] = list(df.columns)
            result["sample"] = df.head(sample_rows).to_dict(orient="records")
        except Exception as exc:
            result["sample_error"] = str(exc)

    return result


@tool_metadata(
    tags={"predictive", "data", "read", "datastore", "list", "daria"},
    description=(
        "[Datastore—list connections] Use when the user works with saved external connections "
        "(DB, warehouse, bucket, etc.) and needs datastore ids. Read-only. Not AI Catalog "
        "datasets (list_ai_catalog_items), not modeling projects. Next step: browse_datastore "
        "or query_datastore."
    ),
)
async def list_datastores() -> dict[str, Any]:
    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    datastores = client.DataStore.list()

    return {
        "datastores": [
            {
                "id": ds.id,
                "canonical_name": getattr(ds, "canonical_name", ""),
                "creator_id": getattr(ds, "creator_id", ""),
                "params": _serialize_datastore_params(getattr(ds, "params", None)),
            }
            for ds in datastores
        ],
        "count": len(datastores),
    }


@tool_metadata(
    tags={"predictive", "data", "read", "datastore", "browse", "daria"},
    description=(
        "[Datastore—browse objects] Use after list_datastores when the user needs schemas, "
        "tables, or folders inside one connection. Read-only; optional path, search filter, "
        "pagination. Not SQL execution (query_datastore), not catalog dataset listing."
    ),
)
async def browse_datastore(
    *,
    datastore_id: Annotated[str, "Connection id from list_datastores."] | None = None,
    path: Annotated[str, "Path within the connection (e.g. schema or folder); omit for root."]
    | None = None,
    offset: Annotated[int, "Pagination offset."] = 0,
    limit: Annotated[int, "Max entries to return."] = 100,
    search: Annotated[str, "Optional filter substring for object names."] | None = None,
) -> dict[str, Any]:
    if not datastore_id:
        raise ToolError("Datastore ID must be provided")

    token = await get_datarobot_access_token()
    dr_module = DataRobotClient(token).get_client()
    rest_client = dr_module.client.get_client()

    params: dict = {"offset": offset, "limit": limit}
    if path:
        params["path"] = path
    if search:
        params["search"] = search
    response = rest_client.get(f"externalDataDrivers/{datastore_id}/tables/", params=params)
    data = response.json()
    items = data.get("data", data) if isinstance(data, dict) else data

    return {
        "datastore_id": datastore_id,
        "path": path or "/",
        "items": items,
        "count": len(items),
    }


@tool_metadata(
    tags={"predictive", "data", "read", "write", "delete", "datastore", "query", "sql", "daria"},
    description=(
        "[Datastore—run SQL] Use when the user wants to execute SQL (SELECT or DML) against "
        "one saved external connection. Requires datastore_id from list_datastores; returns "
        "rows, columns, and row count up to limit. Not DataRobot catalog inspection "
        "(get_dataset_details), not browsing without SQL (browse_datastore)."
    ),
)
async def query_datastore(
    *,
    datastore_id: Annotated[str, "Connection id from list_datastores."] | None = None,
    sql: Annotated[str, "SQL statement to run against that connection."] | None = None,
    limit: Annotated[int, "Max rows returned from the query result."] = 1000,
) -> dict[str, Any]:
    if not datastore_id:
        raise ToolError("Datastore ID must be provided")
    if not sql:
        raise ToolError("SQL query must be provided")

    token = await get_datarobot_access_token()
    dr_module = DataRobotClient(token).get_client()
    rest_client = dr_module.client.get_client()

    payload = {"query": sql, "limit": limit}
    response = rest_client.post(f"externalDataDrivers/{datastore_id}/execute/", json=payload)
    data = response.json()

    return {
        "rows": data.get("data", []),
        "row_count": len(data.get("data", [])),
        "columns": data.get("columns", []),
    }


# from fastmcp import Context

# from datarobot_genai.drmcp.core.memory_management import MemoryManager, get_memory_manager


# @tool_metadata()
# async def list_ai_catalog_items(
#     ctx: Context, agent_id: str = None, storage_id: str = None
# ) -> str:
#     """
#     List all AI Catalog items (datasets) for the authenticated user.

#     Returns:
#         a resource id that can be used to retrieve the list of AI Catalog items using the
#         get_resource tool
#     """
#     token = await get_datarobot_access_token()
#     client = DataRobotClient(token).get_client()
#     datasets = client.Dataset.list()
#     if not datasets:
#         logger.info("No AI Catalog items found")
#         return "No AI Catalog items found."
#     result = "\n".join(f"{ds.id}: {ds.name}" for ds in datasets)

#     if MemoryManager.is_initialized():
#         resource_id = await get_memory_manager().store_resource(
#             data=result,
#             memory_storage_id=storage_id,
#             agent_identifier=agent_id,
#         )
#     else:
#         raise ValueError("MemoryManager is not initialized")

#     logger.info(f"Found {len(datasets)} AI Catalog items")
#     return resource_id
