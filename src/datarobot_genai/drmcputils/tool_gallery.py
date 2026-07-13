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

from typing import Any

from datarobot_genai.drmcputils.categories import categories_for_tool

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


# ``auth_provider`` → OAuth identity ``provider_type``, matching the drtools OAuth token
# lookup (``get_oauth_access_token_with_header_fallback(<provider_type>)``). The connector
# packages are OAuth-first, so they carry an OAuth provider type; the web-search packages
# (perplexity, tavily) authenticate with an API key and have no OAuth provider (→ null).
_OAUTH_PROVIDER_TYPES: dict[str, str] = {
    "jira": "jira",
    "confluence": "confluence",
    "gdrive": "google",
    "microsoft_graph": "microsoft",
}


# Hosted (dynamic) tools are classified by their ``meta.tool_category`` marker — set by the
# tool providers — rather than by the static taxonomy. Each kind maps to the gallery
# ``provider`` and ``categories`` it should report:
#   - USER_TOOL_DEPLOYMENT: DataRobot deployment tools (CustomModelToolProvider). They are
#     DataRobot-served, so ``provider = datarobot``; bucketed under ``dr_dynamic_tools``.
#   - PROXIED_USER_MCP: tools proxied from a user's own MCP server (UserMCPProvider). They are
#     served outside the DataRobot API, so ``provider = third_party``; bucketed under
#     ``dr_proxied_user_mcp``.
_HOSTED_TOOL_KINDS: dict[str, dict[str, str]] = {
    "USER_TOOL_DEPLOYMENT": {"provider": "datarobot", "category": "dr_dynamic_tools"},
    "PROXIED_USER_MCP": {"provider": "third_party", "category": "dr_proxied_user_mcp"},
}


def hosted_kind(tool_category: str | None) -> dict[str, str] | None:
    """Return the gallery classification for a hosted tool, or None if it isn't hosted."""
    if not tool_category:
        return None
    return _HOSTED_TOOL_KINDS.get(tool_category)


def _tool_category(tool: Any) -> str | None:
    """Read the provider's ``meta.tool_category`` marker (None for static drtools tools)."""
    meta = getattr(tool, "meta", None) or {}
    return meta.get("tool_category")


def is_hosted(tool: Any) -> bool:
    """Return True for dynamic/proxied tools — those carrying a hosted ``tool_category``."""
    return hosted_kind(_tool_category(tool)) is not None


def merge_tool_info(tool: Any, ui_metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Combine a FastMCP ``Tool`` with drtools UI metadata + derived categories.

    Carries the raw ``tool_category`` marker and the tool's own ``description`` so the
    builder can classify hosted tools (provider/categories) and fall back to the MCP
    description when there is no curated UI copy.
    """
    ui = ui_metadata.get(tool.name, {})
    return {
        "name": tool.name,
        "display_name": ui.get("display_name"),
        "description_ui": ui.get("description_ui"),
        "description": getattr(tool, "description", None),
        "auth_provider": ui.get("auth_provider"),
        "tags": sorted(tool.tags or []),
        "categories": categories_for_tool(tool.name),
        "tool_category": _tool_category(tool),
        "hosted": is_hosted(tool),
    }


def _provider_for(auth_provider: str | None) -> str:
    """Classify a tool's provider as ``datarobot`` or ``third_party``.

    Third-party = served from outside the DataRobot API: the connector / web-search
    packages (jira, confluence, gdrive, microsoft_graph, perplexity, tavily — the ones
    carrying an ``auth_provider``). External proxied HTTP MCPs (not supported yet) would
    also be third-party once added. Everything else — DataRobot-native tools and DataRobot
    dynamic deployments — is ``datarobot``.
    """
    if auth_provider:
        return "third_party"
    return "datarobot"


def _oauth_provider_type_for(auth_provider: str | None) -> str | None:
    """Describe a tool's third-party auth type.

    - OAuth-first connectors → their OAuth ``provider_type`` (jira/confluence as-is,
      gdrive→google, microsoft_graph→microsoft).
    - Other third parties (perplexity, tavily) → ``"api_key"``.
    - DataRobot-native tools (no ``auth_provider``) → ``None`` (no separate credential).
    """
    if not auth_provider:
        return None
    return _OAUTH_PROVIDER_TYPES.get(auth_provider, "api_key")


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
    items: list[dict] = []
    for t in tools:
        kind = hosted_kind(t.get("tool_category"))
        if kind is not None:
            # Hosted (dynamic) tool: classification comes from its provider's meta marker,
            # not the static taxonomy or a drtools auth_provider.
            provider = kind["provider"]
            oauth_provider_type = None
            categories = [kind["category"]]
            hosted = True
        else:
            auth_provider = t.get("auth_provider")
            provider = _provider_for(auth_provider)
            oauth_provider_type = _oauth_provider_type_for(auth_provider)
            categories = list(t.get("categories") or [])
            hosted = bool(t.get("hosted", False))
        items.append(
            {
                "name": t["name"],
                "display_name": t.get("display_name") or t["name"],
                # Prefer the curated UI copy (drtools ``description_ui``); fall back to the
                # tool's own MCP description (the only copy dynamic/proxied tools carry).
                "description": t.get("description_ui") or t.get("description") or "",
                "tags": sorted(t.get("tags") or []),
                "categories": categories,
                "provider": provider,
                "oauth_provider_type": oauth_provider_type,
                "hosted": hosted,
            }
        )
    return items
