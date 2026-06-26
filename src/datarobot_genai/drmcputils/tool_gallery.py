# Copyright 2026 DataRobot, Inc.
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

"""Tool gallery utilities — shared between MCP servers and the ARD catalog.

This module has no fastmcp dependency and no MCP-protocol imports so it can be
imported by drtools, drmcputils, and drmcpbase alike.
"""

# Keys present in @tool_metadata(...) that carry UI/gallery metadata or
# server-side registration hints.  These must be stripped before the metadata
# dict is forwarded to FastMCP's mcp.tool() call so agents / LLMs never see
# them in tools/list or tools/call responses.
DRTOOLS_PRIVATE_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "excluded_from_global_mcp",
        "display_name",
        "description_ui",
        "auth_provider",
        "categories",
    }
)


def build_tool_gallery_items(tools: list[dict]) -> list[dict]:
    """Build the tools-gallery JSON response items from merged tool dicts.

    Each dict in *tools* should contain at minimum ``name`` and ``hosted``.
    All other keys are optional and fall back to safe defaults.

    Args:
        tools: List of dicts with merged FastMCP tool attrs + drtools metadata.

    Returns
    -------
        Serialisable list of tool gallery item dicts.
    """
    return [
        {
            "name": t["name"],
            "display_name": t.get("display_name") or t["name"],
            "description_ui": t.get("description_ui") or "",
            "tags": sorted(t.get("tags") or []),
            "categories": list(t.get("categories") or []),
            "auth_provider": t.get("auth_provider") or None,
            "hosted": bool(t.get("hosted", False)),
        }
        for t in tools
    ]
