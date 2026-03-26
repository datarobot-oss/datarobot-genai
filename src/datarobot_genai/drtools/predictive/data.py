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

import logging
import os
from typing import Annotated
from typing import Any

from datarobot_genai.drmcp import dr_mcp_integration_tool
from datarobot_genai.drmcp.core.utils import is_valid_url
from datarobot_genai.drtools.core.clients.datarobot import DataRobotClient
from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.exceptions import ToolError

logger = logging.getLogger(__name__)


@dr_mcp_integration_tool(tags={"predictive", "data", "write", "upload", "catalog", "daria"})
async def upload_dataset_to_ai_catalog(
    *,
    file_path: Annotated[str, "The path to the dataset file to upload."] | None = None,
    file_url: Annotated[str, "The URL to the dataset file to upload."] | None = None,
) -> dict[str, Any]:
    """Upload a dataset to the DataRobot AI Catalog / Data Registry."""
    if not file_path and not file_url:
        raise ToolError("Either file_path or file_url must be provided.")
    if file_path and file_url:
        raise ToolError("Please provide either file_path or file_url, not both.")

    # Get client
    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    catalog_item = None
    # If file path is provided, create dataset from file.
    if file_path:
        # Does file exist?
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            raise ToolError(f"File not found: {file_path}")
        catalog_item = client.Dataset.create_from_file(file_path)
    else:
        # Does URL exist?
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


@dr_mcp_integration_tool(tags={"predictive", "data", "read", "list", "catalog", "daria"})
async def list_ai_catalog_items() -> dict[str, Any]:
    """List all AI Catalog items (datasets) for the authenticated user."""
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


@dr_mcp_integration_tool(tags={"predictive", "data", "read", "dataset", "metadata", "daria"})
async def get_dataset_details(
    *,
    dataset_id: Annotated[str, "The ID of the DataRobot dataset"] | None = None,
    include_sample: Annotated[bool, "Whether to include sample rows"] = True,
    sample_rows: Annotated[int, "Number of sample rows to return"] = 10,
) -> dict[str, Any]:
    """Get DataRobot dataset metadata and optional sample rows."""
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


@dr_mcp_integration_tool(tags={"predictive", "data", "read", "datastore", "list", "daria"})
async def list_datastores() -> dict[str, Any]:
    """List available DataRobot data connections (datastores)."""
    token = await get_datarobot_access_token()
    client = DataRobotClient(token).get_client()
    datastores = client.DataStore.list()

    return {
        "datastores": [
            {
                "id": ds.id,
                "canonical_name": getattr(ds, "canonical_name", ""),
                "creator_id": getattr(ds, "creator_id", ""),
                "params": getattr(ds, "params", {}),
            }
            for ds in datastores
        ],
        "count": len(datastores),
    }


@dr_mcp_integration_tool(tags={"predictive", "data", "read", "datastore", "browse", "daria"})
async def browse_datastore(
    *,
    datastore_id: Annotated[str, "The ID of the datastore to browse"] | None = None,
    path: Annotated[str, "The path to browse within the datastore"] | None = None,
    offset: Annotated[int, "Pagination offset"] = 0,
    limit: Annotated[int, "Maximum number of items to return"] = 100,
    search: Annotated[str, "Search filter for items"] | None = None,
) -> dict[str, Any]:
    """Browse a DataRobot data connection to list catalogs, schemas, and tables."""
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


@dr_mcp_integration_tool(
    tags={"predictive", "data", "read", "write", "delete", "datastore", "query", "sql", "daria"}
)
async def query_datastore(
    *,
    datastore_id: Annotated[str, "The ID of the datastore to query"] | None = None,
    sql: Annotated[str, "The SQL query to execute"] | None = None,
    limit: Annotated[int, "Maximum number of rows to return"] = 1000,
) -> dict[str, Any]:
    """Execute a SQL query against a DataRobot datastore connection.

    Only data manipulation language queries (insert, update, and delete data)
    are supported — no commits or rollbacks.
    """
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


# @dr_mcp_integration_tool()
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
