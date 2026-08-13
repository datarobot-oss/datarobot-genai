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

"""ARD (Agentic Resource Discovery) catalog for global-mcp.

Exposes an importable function that returns the pre-built global-mcp tool catalog — tool
names, their categories, and the MCP endpoint URL — as plain JSON. Agents use it to discover
what global-mcp offers *without* connecting to the server.

The catalog reflects the tools global-mcp **actually registers**, derived from the single
source of truth in :mod:`datarobot_genai.drmcputils.global_mcp_tools`
(``GLOBAL_MCP_PACKAGE_CATEGORIES``) — the same structure the global-mcp registry reads to decide
which packages to load. So ARD and registration cannot drift: enabling/disabling a package or
excluding a tool is a one-line edit there, reflected in both. Today's enabled surface is
predictive (catalog / modeling / deployments / predictions), use_case, perplexity, tavily,
dr_docs, vdb, files_api, and workload; connectors, panels, hosted/dynamic categories, and the
``dr_mcpapps`` placeholder are not registered, so they are not advertised. ``file_upload`` is
excluded from files_api (needs local disk access).

A cross-check test in global-mcp asserts this catalog matches ``register_all_drtools`` output,
so a taxonomy/rename mismatch still fails CI.

This module imports only ``drmcputils`` — no fastmcp, no drtools — so it stays in the lowest
layer and is cheap to import from anywhere.
"""

from datarobot_genai.drmcputils.categories import LEAF_CATEGORY_TOOLS
from datarobot_genai.drmcputils.global_mcp_tools import GLOBAL_MCP_EXCLUDED_TOOLS
from datarobot_genai.drmcputils.global_mcp_tools import global_mcp_leaf_categories

# Default production endpoint for the global-mcp server.
GLOBAL_MCP_URL = "https://app.datarobot.com/api/v2/genai/globalmcp/mcp"


def get_global_mcp_prebuilt_tools(mcp_url: str = GLOBAL_MCP_URL) -> dict:
    """Return the pre-built global-mcp tool catalog for ARD.

    Args:
        mcp_url: Override the MCP endpoint URL.

    Returns
    -------
        ``{"mcp_url": str, "tools": [{"name": str, "category": str}, ...], "count": int}`` —
        the tools global-mcp registers (see module docstring), sorted by (category, name).
    """
    tools: list[dict[str, str]] = []
    for category in sorted(global_mcp_leaf_categories()):
        for name in sorted(LEAF_CATEGORY_TOOLS[category] - GLOBAL_MCP_EXCLUDED_TOOLS):
            tools.append({"name": name, "category": str(category)})

    return {"mcp_url": mcp_url, "tools": tools, "count": len(tools)}
