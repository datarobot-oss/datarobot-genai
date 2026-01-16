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


from fastmcp.tools import Tool
from mcp.types import Tool as MCPTool


def filter_tools_by_tags(
    tools: list[Tool | MCPTool],
    tags: list[str] | None = None,
    match_all: bool = False,
) -> list[Tool | MCPTool]:
    """
    Filter tools by tags.

    Args:
        tools: List of tools to filter
        tags: List of tags to filter by. If None, returns all tools
        match_all: If True, tool must have all specified tags. If False, tool must have at least
            one tag.

    Returns
    -------
        List of tools that match the tag criteria
    """
    if not tags:
        return tools

    filtered_tools = []

    for tool in tools:
        tool_tags = get_tool_tags(tool)

        if not tool_tags:
            continue

        if match_all:
            # Tool must have all specified tags
            if all(tag in tool_tags for tag in tags):
                filtered_tools.append(tool)
        elif any(tag in tool_tags for tag in tags):
            # Tool must have at least one specified tag
            filtered_tools.append(tool)

    return filtered_tools


def get_tool_tags(tool: Tool | MCPTool) -> list[str]:
    """
    Get tags for a specific tool.

    Args:
        tool: The tool to get tags for

    Returns
    -------
        List of tags for the tool
    """
    # Primary: native FastMCP meta location
    if hasattr(tool, "meta") and getattr(tool, "meta"):
        fastmcp_meta = tool.meta.get("_fastmcp", {})
        meta_tags = fastmcp_meta.get("tags", [])
        if isinstance(meta_tags, list):
            return meta_tags

    # Fallback: annotations.tags (for compatibility during transition)
    if tool.annotations and hasattr(tool.annotations, "tags"):
        tags = getattr(tool.annotations, "tags", [])
        return tags if isinstance(tags, list) else []

    return []


def list_all_tags(tools: list[Tool | MCPTool]) -> list[str]:
    """
    Get all unique tags from a list of tools.

    Args:
        tools: List of tools to extract tags from

    Returns
    -------
        List of unique tags
    """
    all_tags = set()
    for tool in tools:
        tool_tags = get_tool_tags(tool)
        all_tags.update(tool_tags)

    return sorted(list(all_tags))


def get_tools_by_tag(tools: list[Tool | MCPTool], tag: str) -> list[Tool | MCPTool]:
    """
    Get all tools that have a specific tag.

    Args:
        tools: List of tools to search
        tag: The tag to search for

    Returns
    -------
        List of tools with the specified tag
    """
    return filter_tools_by_tags(tools, [tag])
