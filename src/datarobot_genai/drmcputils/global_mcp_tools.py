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

"""Single source of truth for the tools global-mcp exposes.

Both consumers read from here, so they cannot drift:

- the global-mcp registry (``register_all_drtools``) — which drtools *packages* to load;
- the ARD catalog (``get_global_mcp_prebuilt_tools``) — which *tools* to advertise.

``GLOBAL_MCP_PACKAGE_CATEGORIES`` replaces the per-package ``GLOBAL_MCP_ENABLED`` flags that
used to live scattered across the drtools ``__init__`` files. Enabling a package for global-mcp
is now a one-line edit here: add its drtools package name mapped to the leaf categories it
contributes. ``GLOBAL_MCP_EXCLUDED_TOOLS`` is a per-tool safety net — names listed there are
never registered and never advertised, even if their package/category is enabled.

Lives in ``drmcputils`` (the lowest, fastmcp-free layer) beside the taxonomy it derives from,
so every layer above — and the global-mcp registry — can import it. Imports only the taxonomy.
"""

from datarobot_genai.drmcputils.categories import MCPToolCategory

# ── SINGLE SOURCE OF TRUTH ───────────────────────────────────────────────────
# drtools package (directory name under ``datarobot_genai.drtools``) → the leaf categories
# it contributes. The keys are what global-mcp registers; the values drive what ARD advertises.
GLOBAL_MCP_PACKAGE_CATEGORIES: dict[str, frozenset[MCPToolCategory]] = {
    "predictive": frozenset(
        {
            MCPToolCategory.DR_CATALOG,
            MCPToolCategory.DR_MODELING,
            MCPToolCategory.DR_DEPLOYMENTS,
            MCPToolCategory.DR_PREDICTIONS,
        }
    ),
    "use_case": frozenset({MCPToolCategory.DR_USE_CASES}),
    "perplexity": frozenset({MCPToolCategory.DR_WEB_SEARCH_PERPLEXITY}),
    "tavily": frozenset({MCPToolCategory.DR_WEB_SEARCH_TAVILY}),
    "dr_docs": frozenset({MCPToolCategory.DR_DOCUMENTATION}),
    "vdb": frozenset({MCPToolCategory.DR_VDB}),
}

# Per-tool opt-out applied on top of the enabled packages — the single place a specific tool is
# blocked from being registered *and* advertised. This is the one source of truth for tool
# exclusions (there is no per-tool metadata flag anymore).
#   - file_upload: needs local disk access, so it must never be exposed by global-mcp. Listed
#     now even though its package (files_api) isn't enabled yet, so it stays blocked if/when it is.
GLOBAL_MCP_EXCLUDED_TOOLS: frozenset[str] = frozenset({"file_upload"})


def global_mcp_enabled_packages() -> frozenset[str]:
    """Return the drtools package names global-mcp registers."""
    return frozenset(GLOBAL_MCP_PACKAGE_CATEGORIES)


def global_mcp_leaf_categories() -> frozenset[MCPToolCategory]:
    """Return every leaf category contributed by the enabled packages."""
    return frozenset().union(*GLOBAL_MCP_PACKAGE_CATEGORIES.values())
