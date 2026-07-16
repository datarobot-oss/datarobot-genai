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

"""Tool gallery: data models, static config, visibility, and filtering.

New module per tech spec §2.  Static configs (category/provider mappings,
tool-auth mapping) are defined here and loaded once at startup.

Note: ``GalleryTool`` is used throughout implementation code to avoid
colliding with ``fastmcp.tools.Tool``.  The REST response uses the camelCase
alias ``Tool`` defined in the spec.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from fastmcp.tools import Tool as FastMCPTool

from datarobot_genai.drmcp.core.utils import get_tool_tags

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ToolCategory(str, Enum):
    DATA_CONNECTORS = "data_connectors"
    WEB_SEARCH = "web_search"
    SOFTWARE_DEV_DEVOPS = "software_dev_devops"
    DATA_VISUALIZATION = "data_visualization"
    PREDICTIVE = "predictive"


class ToolProviderName(str, Enum):
    DATAROBOT = "datarobot"
    THIRD_PARTY = "third_party"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ToolProvider:
    name: ToolProviderName
    is_third_party: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "isThirdParty": self.is_third_party,
        }


@dataclass
class GalleryTool:
    """REST response model for a gallery item.

    Aliased as ``Tool`` on the wire (camelCase JSON).
    """

    name: str
    description: str
    tags: list[str]
    category: ToolCategory
    provider: ToolProvider
    dr_oauth_provider_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "category": self.category.value,
            "provider": self.provider.to_dict(),
            "drOAuthProviderType": self.dr_oauth_provider_type,
        }


# ---------------------------------------------------------------------------
# Static config: prefix → (category, provider, dr_oauth_provider_type)
# Tech spec §2.3 and §1.6
# ---------------------------------------------------------------------------

@dataclass
class _PrefixConfig:
    category: ToolCategory
    provider_name: ToolProviderName
    is_third_party: bool
    dr_oauth_provider_type: str | None = None


# Ordered: longer/more-specific prefixes first so longest-match wins.
_PREFIX_CONFIGS: list[tuple[str, _PrefixConfig]] = [
    # Data connectors — third-party
    ("dr_connector_confluence", _PrefixConfig(ToolCategory.DATA_CONNECTORS, ToolProviderName.THIRD_PARTY, True, "confluence")),
    ("dr_connector_jira", _PrefixConfig(ToolCategory.DATA_CONNECTORS, ToolProviderName.THIRD_PARTY, True, "jira")),
    ("dr_connector_gdrive", _PrefixConfig(ToolCategory.DATA_CONNECTORS, ToolProviderName.THIRD_PARTY, True, "google")),
    ("dr_connector_microsoft_sharepoint_onedrive", _PrefixConfig(ToolCategory.DATA_CONNECTORS, ToolProviderName.THIRD_PARTY, True, "microsoft")),
    ("dr_connector_", _PrefixConfig(ToolCategory.DATA_CONNECTORS, ToolProviderName.THIRD_PARTY, True)),
    # Web search — third-party
    ("dr_web_search_perplexity", _PrefixConfig(ToolCategory.WEB_SEARCH, ToolProviderName.THIRD_PARTY, True)),
    ("dr_web_search_tavily", _PrefixConfig(ToolCategory.WEB_SEARCH, ToolProviderName.THIRD_PARTY, True)),
    ("dr_web_search_", _PrefixConfig(ToolCategory.WEB_SEARCH, ToolProviderName.THIRD_PARTY, True)),
    # Predictive — DataRobot
    ("dr_use_cases", _PrefixConfig(ToolCategory.PREDICTIVE, ToolProviderName.DATAROBOT, False)),
    ("dr_catalog", _PrefixConfig(ToolCategory.PREDICTIVE, ToolProviderName.DATAROBOT, False)),
    ("dr_modeling", _PrefixConfig(ToolCategory.PREDICTIVE, ToolProviderName.DATAROBOT, False)),
    ("dr_deployments", _PrefixConfig(ToolCategory.PREDICTIVE, ToolProviderName.DATAROBOT, False)),
    ("dr_predictions", _PrefixConfig(ToolCategory.PREDICTIVE, ToolProviderName.DATAROBOT, False)),
    # Software dev / DevOps — DataRobot
    ("dr_workload", _PrefixConfig(ToolCategory.SOFTWARE_DEV_DEVOPS, ToolProviderName.DATAROBOT, False)),
    ("dr_file", _PrefixConfig(ToolCategory.SOFTWARE_DEV_DEVOPS, ToolProviderName.DATAROBOT, False)),
    # Data visualisation — DataRobot
    ("dr_mcpapps", _PrefixConfig(ToolCategory.DATA_VISUALIZATION, ToolProviderName.DATAROBOT, False)),
    ("dr_panels", _PrefixConfig(ToolCategory.DATA_VISUALIZATION, ToolProviderName.DATAROBOT, False)),
]

# Sentinel for deployment-backed tools (tool meta tool_category = USER_TOOL_DEPLOYMENT)
_DEPLOYMENT_TOOL_CONFIG = _PrefixConfig(ToolCategory.PREDICTIVE, ToolProviderName.DATAROBOT, False)

# Default fallback for tools that match no prefix
_DEFAULT_CONFIG = _PrefixConfig(ToolCategory.SOFTWARE_DEV_DEVOPS, ToolProviderName.DATAROBOT, False)


def _resolve_prefix_config(module_name: str) -> _PrefixConfig:
    """Return the config for *module_name* using longest-prefix match."""
    for prefix, cfg in _PREFIX_CONFIGS:
        if module_name.startswith(prefix):
            return cfg
    return _DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

# Internal tool_category values stored in tool.meta
_BUILT_IN_TOOL = "BUILT_IN_TOOL"
_USER_TOOL = "USER_TOOL"
_USER_TOOL_DEPLOYMENT = "USER_TOOL_DEPLOYMENT"


def _get_tool_category(tool: FastMCPTool) -> str:
    """Extract the internal tool_category from tool meta."""
    meta = getattr(tool, "meta", None) or {}
    if isinstance(meta, dict):
        return meta.get("tool_category", _BUILT_IN_TOOL)
    return _BUILT_IN_TOOL


def _get_tool_created_by(tool: FastMCPTool) -> str | None:
    """Extract the created_by user ID from tool meta."""
    meta = getattr(tool, "meta", None) or {}
    if isinstance(meta, dict):
        return meta.get("created_by")
    return None


def is_tool_visible_to_caller(tool: FastMCPTool, *, caller_created_by: str | None) -> bool:
    """Return True when *tool* is visible to the caller.

    Visibility rules (§2.2):
    - Built-in / platform tools: always visible.
    - User-owned tools: visible when created_by matches caller.
    - Deployment-backed tools: visible when created_by matches caller.
      (In production this would also check deployment access; for POC we use
      created_by ownership as a proxy.)
    """
    tool_category = _get_tool_category(tool)

    if tool_category == _BUILT_IN_TOOL:
        return True

    if caller_created_by is None:
        # No identity available — only built-ins are visible.
        return False

    tool_created_by = _get_tool_created_by(tool)
    if tool_category in (_USER_TOOL, _USER_TOOL_DEPLOYMENT):
        return tool_created_by == caller_created_by

    # Unknown category — treat as built-in.
    return True


# ---------------------------------------------------------------------------
# Building gallery tool objects
# ---------------------------------------------------------------------------


def build_tool(mcp_tool: FastMCPTool, *, caller_created_by: str | None = None) -> GalleryTool:
    """Build a :class:`GalleryTool` from a FastMCP tool.

    Uses the tool's module prefix for category/provider resolution, falling
    back to the deployment-tool config when tool_category is USER_TOOL_DEPLOYMENT.
    """
    tool_category = _get_tool_category(mcp_tool)

    if tool_category == _USER_TOOL_DEPLOYMENT:
        cfg = _DEPLOYMENT_TOOL_CONFIG
    else:
        fn = getattr(mcp_tool, "fn", None)
        module_name = getattr(fn, "__module__", "") or ""
        # Strip the package prefix to get the tool-family name.
        # e.g. "datarobot_genai.drtools.dr_connector_jira.tools" → "dr_connector_jira"
        parts = module_name.split(".")
        # Find the first segment starting with "dr_"
        tool_module = next((p for p in parts if p.startswith("dr_")), module_name)
        cfg = _resolve_prefix_config(tool_module)

    tags = sorted(get_tool_tags(mcp_tool))

    return GalleryTool(
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        tags=tags,
        category=cfg.category,
        provider=ToolProvider(name=cfg.provider_name, is_third_party=cfg.is_third_party),
        dr_oauth_provider_type=cfg.dr_oauth_provider_type,
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_gallery_tools(
    items: list[GalleryTool],
    *,
    name: str | None = None,
    tag: str | None = None,
    ownership: str | None = None,
    category: list[str] | None = None,
    provider: list[str] | None = None,
    caller_created_by: str | None = None,
    original_tools: list[FastMCPTool] | None = None,
) -> list[GalleryTool]:
    """Apply query-param filters to an already-visibility-scoped list.

    Args:
        items: Pre-built ``GalleryTool`` list (already visibility-filtered).
        name: Exact match on tool name.
        tag: Match-any single tag.
        ownership: ``"owned_by_me"`` to exclude built-ins and keep only
            caller-owned tools.
        category: Match-any list of ``ToolCategory`` values.
        provider: Match-any list of ``ToolProviderName`` values.
        caller_created_by: Required when *ownership* is set.
        original_tools: Parallel list of raw ``FastMCPTool`` objects used for
            ``ownership`` filtering (to inspect ``tool_category``).
    """
    result = items
    orig = original_tools or []

    if name is not None:
        result = [t for t in result if t.name == name]

    if tag is not None:
        result = [t for t in result if tag in t.tags]

    if category:
        cat_values = set(category)
        result = [t for t in result if t.category.value in cat_values]

    if provider:
        prov_values = set(provider)
        result = [t for t in result if t.provider.name.value in prov_values]

    if ownership == "owned_by_me":
        # Build a set of names that belong to the caller (exclude built-ins).
        if caller_created_by and orig:
            owned_names = {
                mcp_t.name
                for mcp_t in orig
                if _get_tool_category(mcp_t) in (_USER_TOOL, _USER_TOOL_DEPLOYMENT)
                and _get_tool_created_by(mcp_t) == caller_created_by
            }
            result = [t for t in result if t.name in owned_names]
        else:
            result = []

    return result
