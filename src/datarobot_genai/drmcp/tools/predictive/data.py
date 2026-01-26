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

import json
import logging
import os
from typing import Annotated

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool
from datarobot_genai.drmcp.core.utils import is_valid_url

logger = logging.getLogger(__name__)


@dr_mcp_tool(tags={"predictive", "data", "write", "upload", "catalog"})
async def upload_dataset_to_ai_catalog(
    *,
    file_path: Annotated[str, "The path to the dataset file to upload."] | None = None,
    file_url: Annotated[str, "The URL to the dataset file to upload."] | None = None,
) -> ToolError | ToolResult:
    """Upload a dataset to the DataRobot AI Catalog / Data Registry."""
    if not file_path and not file_url:
        raise ToolError("Either file_path or file_url must be provided.")
    if file_path and file_url:
        raise ToolError("Please provide either file_path or file_url, not both.")

    # Get client
    client = get_sdk_client()
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

    return ToolResult(
        content=f"Successfully uploaded dataset: {catalog_item.id}",
        structured_content={
            "dataset_id": catalog_item.id,
            "dataset_version_id": catalog_item.version_id,
            "dataset_name": catalog_item.name,
        },
    )


@dr_mcp_tool(tags={"predictive", "data", "read", "list", "catalog"})
async def list_ai_catalog_items() -> ToolResult:
    """List all AI Catalog items (datasets) for the authenticated user."""
    client = get_sdk_client()
    datasets = client.Dataset.list()

    if not datasets:
        logger.info("No AI Catalog items found")
        return ToolResult(
            content="No AI Catalog items found.",
            structured_content={"datasets": []},
        )

    datasets_dict = {ds.id: ds.name for ds in datasets}
    datasets_count = len(datasets)

    return ToolResult(
        content=(
            f"Found {datasets_count} AI Catalog items, here are the details:\n"
            f"{json.dumps(datasets_dict, indent=2)}"
        ),
        structured_content={
            "datasets": datasets_dict,
            "count": datasets_count,
        },
    )


# from fastmcp import Context

# from datarobot_genai.drmcp.core.memory_management import MemoryManager, get_memory_manager


# @dr_mcp_tool()
# async def list_ai_catalog_items(
#     ctx: Context, agent_id: str = None, storage_id: str = None
# ) -> str:
#     """
#     List all AI Catalog items (datasets) for the authenticated user.

#     Returns:
#         a resource id that can be used to retrieve the list of AI Catalog items using the
#         get_resource tool
#     """
#     client = get_sdk_client()
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
