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

from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_tool

logger = logging.getLogger(__name__)


@dr_mcp_tool(tags={"predictive", "data", "write", "upload", "catalog"})
async def upload_dataset_to_ai_catalog(
    file_path: Annotated[str, "The path to the dataset file to upload."],
) -> ToolResult:
    """Upload a dataset to the DataRobot AI Catalog."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise ToolError(f"File not found: {file_path}")

    client = get_sdk_client()
    catalog_item = client.Dataset.create_from_file(file_path)

    return ToolResult(
        content=f"Successfully uploaded dataset: {catalog_item.id}.",
        structured_content={
            "id": catalog_item.id,
            "name": catalog_item.name,
            "status": catalog_item.status,
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

    return ToolResult(
        content=f"Found {len(datasets)} AI Catalog items.",
        structured_content={
            "datasets": [{"id": ds.id, "name": ds.name} for ds in datasets],
            "count": len(datasets),
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
