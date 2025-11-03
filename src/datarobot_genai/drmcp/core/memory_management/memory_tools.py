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
from typing import Any

from datarobot_genai.drmcp.core.mcp_instance import dr_core_mcp_tool

from .manager import ToolContext
from .manager import get_memory_manager


@dr_core_mcp_tool()
async def store_resource(
    data: Any,
    memory_storage_id: str | None = None,
    agent_identifier: str | None = None,
    prompt: str | None = None,
    tool_name: str | None = None,
    tool_parameters: dict[str, Any] | None = None,
    embedding_vector: list[float] | None = None,
) -> str:
    """
    Store a resource in the memory storage.

    Args:
        data: The data to store (string, json or binary)
        memory_storage_id: Optional storage ID to associate the resource with
        agent_identifier: Required if memory_storage_id is provided
        prompt: Optional prompt used to generate this resource
        tool_name: Optional name of the tool used to generate this resource
        tool_parameters: Optional parameters used with the tool
        embedding_vector: Optional embedding vector for the resource

    Returns
    -------
        str: The ID of the stored resource
    """
    tool_context = None
    if tool_name and tool_parameters:
        tool_context = ToolContext(name=tool_name, parameters=tool_parameters)

    memory_manager = get_memory_manager()
    if not memory_manager:
        return "Memory manager not initialized"

    resource_id = await memory_manager.store_resource(
        data=data,
        memory_storage_id=memory_storage_id,
        agent_identifier=agent_identifier,
        prompt=prompt,
        tool_context=tool_context,
        embedding_vector=embedding_vector,
    )
    return f"Resource stored with ID: {resource_id}"


@dr_core_mcp_tool()
async def get_resource(
    resource_id: str,
    memory_storage_id: str | None = None,
    agent_identifier: str | None = None,
    include_data: bool = True,
) -> str:
    """
    Get a resource and optionally its data from the memory storage.

    Args:
        resource_id: The ID of the resource to retrieve
        memory_storage_id: Optional storage ID the resource belongs to
        agent_identifier: Required if memory_storage_id is provided
        include_data: Whether to include the resource data in the response

    Returns
    -------
        str: JSON string containing the resource metadata and optionally its data
    """
    memory_manager = get_memory_manager()
    if not memory_manager:
        return "Memory manager not initialized"

    resource = await memory_manager.get_resource(
        resource_id=resource_id,
        memory_storage_id=memory_storage_id,
        agent_identifier=agent_identifier,
    )

    if not resource:
        return "Resource not found"

    result = {
        "id": resource.id,
        "memory_storage_id": resource.memory_storage_id,
        "prompt": resource.prompt,
        "tool_context": resource.tool_context.model_dump() if resource.tool_context else None,
        "embedding_vector": resource.embedding_vector,
        "created_at": resource.created_at.isoformat(),
    }

    if include_data:
        data = await memory_manager.get_resource_data(
            resource_id=resource_id,
            memory_storage_id=memory_storage_id,
            agent_identifier=agent_identifier,
        )
        if isinstance(data, bytes):
            try:
                # Try to decode as string if possible
                result["data"] = data.decode("utf-8")
            except UnicodeDecodeError:
                # If binary data, return as is
                result["data"] = data  # type: ignore[assignment]
        else:
            result["data"] = data

    return json.dumps(result, default=str)


@dr_core_mcp_tool()
async def list_resources(agent_identifier: str, memory_storage_id: str | None = None) -> str:
    """
    List all resources from the memory storage.

    Args:
        agent_identifier: Agent identifier to scope the search
        memory_storage_id: Optional Storage ID to filter resources

    Returns
    -------
        str: JSON string containing a list of resources
    """
    memory_manager = get_memory_manager()
    if not memory_manager:
        return "Memory manager not initialized"

    resources = await memory_manager.list_resources(
        agent_identifier=agent_identifier, memory_storage_id=memory_storage_id
    )

    if not resources:
        return "No resources found"

    result = []
    for resource in resources:
        result.append(
            {
                "id": resource.id,
                "memory_storage_id": resource.memory_storage_id,
                "prompt": resource.prompt,
                "tool_context": resource.tool_context.model_dump()
                if resource.tool_context
                else None,
                "created_at": resource.created_at.isoformat(),
            }
        )

    return json.dumps(result, default=str)


@dr_core_mcp_tool()
async def delete_resource(
    resource_id: str,
    memory_storage_id: str | None = None,
    agent_identifier: str | None = None,
) -> str:
    """
    Delete a resource from the memory storage.

    Args:
        resource_id: The ID of the resource to delete
        memory_storage_id: Optional storage ID the resource belongs to
        agent_identifier: Required if memory_storage_id is provided

    Returns
    -------
        str: Success or error message
    """
    memory_manager = get_memory_manager()
    if not memory_manager:
        return "Memory manager not initialized"

    success = await memory_manager.delete_resource(
        resource_id=resource_id,
        memory_storage_id=memory_storage_id,
        agent_identifier=agent_identifier,
    )

    if success:
        return f"Resource {resource_id} deleted successfully"
    return f"Failed to delete resource {resource_id}"
