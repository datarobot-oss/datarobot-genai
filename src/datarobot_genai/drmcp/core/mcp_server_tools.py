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

from .mcp_instance import dr_core_mcp_tool
from .mcp_instance import mcp

logger = logging.getLogger(__name__)


@dr_core_mcp_tool(tags={"mcp_server_tools", "metadata"})
async def get_all_available_tags() -> str:
    """
    List all unique tags from all registered tools.

    Returns
    -------
        A string containing all available tags, one per line.
    """
    tags = await mcp.get_all_tags()
    if not tags:
        return "No tags found in any tools."

    return "\n".join(sorted(tags))


@dr_core_mcp_tool(tags={"mcp_server_tools", "metadata", "discovery"})
async def list_tools_by_tags(tags: list[str] | None = None, match_all: bool = False) -> str:
    """
    List tools filtered by tags.

    Args:
        tags: Optional list of tags to filter by. If None, returns all tools.
        match_all: If True, tool must have all specified tags (AND logic).
                  If False, tool must have at least one tag (OR logic).
                  Only used when tags is provided.

    Returns
    -------
        A formatted string listing tools that match the tag criteria.
    """
    tools = await mcp.list_tools(tags=tags, match_all=match_all)

    if not tools:
        if tags:
            logic = "all" if match_all else "any"
            return f"No tools found with {logic} of the tags: {', '.join(tags)}"
        else:
            return "No tools found."

    result = []
    if tags:
        logic = "all" if match_all else "any"
        result.append(f"Tools with {logic} of the tags: {', '.join(tags)}")
    else:
        result.append("All available tools:")

    result.append("")

    for i, tool in enumerate(tools, 1):
        tool_tags = []
        if tool.annotations and hasattr(tool.annotations, "extra") and tool.annotations.extra:
            tool_tags = tool.annotations.extra.get("tags", [])

        result.append(f"{i}. {tool.name}")
        result.append(f"   Description: {tool.description}")
        if tool_tags:
            result.append(f"   Tags: {', '.join(tool_tags)}")
        result.append("")

    return "\n".join(result)


@dr_core_mcp_tool(tags={"mcp_server_tools", "metadata", "discovery"})
async def get_tool_info_by_name(tool_name: str) -> str:
    """
    Get detailed information about a specific tool by name.

    Args:
        tool_name: The name of the tool to get information about.

    Returns
    -------
        A formatted string with detailed information about the tool.
    """
    all_tools = await mcp.list_tools()

    for tool in all_tools:
        if tool.name == tool_name:
            result = [f"Tool: {tool.name}"]
            result.append(f"Description: {tool.description}")

            # Get tags
            tool_tags = []
            if tool.annotations and hasattr(tool.annotations, "extra") and tool.annotations.extra:
                tool_tags = tool.annotations.extra.get("tags", [])

            if tool_tags:
                result.append(f"Tags: {', '.join(tool_tags)}")
            else:
                result.append("Tags: None")

            # Get input schema info
            if (
                tool.inputSchema
                and hasattr(tool.inputSchema, "properties")
                and tool.inputSchema.properties
            ):
                result.append("Parameters:")
                for param_name, param_info in tool.inputSchema.properties.items():
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "No description")
                    result.append(f"  - {param_name} ({param_type}): {param_desc}")

            return "\n".join(result)

    return f"Tool '{tool_name}' not found."
