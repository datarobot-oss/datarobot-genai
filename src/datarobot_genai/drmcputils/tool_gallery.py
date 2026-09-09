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

"""Tool marker classification — shared by request-time filtering and registration.

This module has no fastmcp dependency and no MCP-protocol imports so it can be
imported by drtools, drmcputils, and drmcpbase alike.

The gallery-*response*-building half of this (``merge_tool_info``,
``build_tool_gallery_items``, ``TOOL_PROVIDER_LABELS``, ...) moved to global-mcp's
``tool_gallery/builders.py`` alongside the ``GET /toolGallery/*`` routes it
serves — global-mcp is now the only server exposing that gallery. What stays here is
the marker-classification genai's own request-time filtering
(``drmcpbase.fastmcp_transforms.utils.is_tool_allowed``) and tool registration
(``drmcp.core.drtools_registry``) depend on independent of any HTTP route.
"""

from typing import Any

# Keys present in @tool_metadata(...) that carry UI/gallery metadata. These must be stripped
# before the metadata dict is forwarded to FastMCP's mcp.tool() call so agents / LLMs never see
# them in tools/list or tools/call responses.
DRTOOLS_PRIVATE_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "display_name",
        "description_ui",
        "auth_provider",
        "categories",
    }
)


# Provider classification reported on each gallery item (see global-mcp's
# ``tool_gallery/builders.py``) and exposed as the ``GET /toolGallery/providers/``
# filter enum.
# ``datarobot`` = served by the DataRobot API.
# ``third_party`` = served from outside it (OAuth / API-key connectors, proxied MCPs).
PROVIDER_DATAROBOT = "datarobot"
PROVIDER_THIRD_PARTY = "third_party"


# Marker-classified tools: registrars stamp ``meta.tool_category`` and each marker maps
# to the gallery ``provider``/``category`` it should report plus whether the tool is
# *hosted* (resolved dynamically at request time rather than statically registered):
#   - USER_TOOL: tools the user authored in their own MCP server code (``dr_mcp_tool``'s
#     default marker). DataRobot-served, NOT hosted; bucketed under ``dr_user_tools``
#     because they live outside the predefined static taxonomy.
#   - USER_TOOL_DEPLOYMENT: DataRobot deployment tools (CustomModelToolProvider). They are
#     DataRobot-served, hosted; bucketed under ``dr_dynamic_tools``.
#   - PROXIED_USER_MCP: tools proxied from a user's own MCP server (UserMCPProvider). They
#     are served outside the DataRobot API, so ``provider = third_party``, hosted. They
#     carry no category (``None``) — the former ``dr_proxied_user_mcp`` bucket was removed.
# BUILT_IN_TOOL deliberately has no row: built-ins are classified by the static taxonomy
# (``categories_for_tool``) and their drtools ``auth_provider``.
_MARKED_TOOL_KINDS: dict[str, dict[str, Any]] = {
    "USER_TOOL": {
        "provider": PROVIDER_DATAROBOT,
        "provider_name": "DataRobot",
        "category": "dr_user_tools",
        "hosted": False,
    },
    "USER_TOOL_DEPLOYMENT": {
        "provider": PROVIDER_DATAROBOT,
        "provider_name": "DataRobot",
        "category": "dr_dynamic_tools",
        "hosted": True,
    },
    # Proxied tools come from a user's own MCP server — no brand name to report.
    "PROXIED_USER_MCP": {
        "provider": PROVIDER_THIRD_PARTY,
        "provider_name": None,
        "category": None,
        "hosted": True,
    },
}


def marked_kind(tool_category: str | None) -> dict[str, Any] | None:
    """Return the gallery classification for a marker-classified tool, or None."""
    if not tool_category:
        return None
    return _MARKED_TOOL_KINDS.get(tool_category)
