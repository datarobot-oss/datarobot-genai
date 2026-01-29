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

"""Dynamic registration of GitHub MCP tools.

This module handles the dynamic registration of GitHub tools from a JSON manifest
at startup. Tools are loaded from the cached manifest file.

Flow:
1. Load tool definitions from github_tools.json manifest
2. Register all enabled tools with dr_mcp_extras()
3. At execution time, tools authenticate via OAuth (get_access_token("github"))

To update the manifest with the latest tools from GitHub's MCP server, run:
    python -m datarobot_genai.drmcp.tools.github.scripts.fetch_manifest
"""

import json
import logging
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from fastmcp.tools.tool import FunctionTool

from datarobot_genai.drmcp.core.mcp_instance import dr_mcp_extras
from datarobot_genai.drmcp.core.mcp_instance import mcp
from datarobot_genai.drmcp.tools.github.tools import call_github_tool

logger = logging.getLogger(__name__)

# Default manifest path
DEFAULT_MANIFEST_PATH = Path(__file__).parent / "github_tools.json"

# Manifest schema version
MANIFEST_SCHEMA = "github_tools_manifest_v1"


def load_manifest(manifest_path: Path | None = None) -> list[dict[str, Any]]:
    """Load tool definitions from the manifest JSON file.

    Args:
        manifest_path: Path to the manifest file. Defaults to github_tools.json.

    Returns
    -------
        List of tool definitions from the manifest

    Raises
    ------
        FileNotFoundError: If manifest file doesn't exist
        json.JSONDecodeError: If manifest file is invalid JSON
    """
    path = manifest_path or DEFAULT_MANIFEST_PATH

    if not path.exists():
        raise FileNotFoundError(f"GitHub tools manifest not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Validate schema
    if data.get("$schema") != MANIFEST_SCHEMA:
        logger.warning(
            f"Manifest schema mismatch: expected {MANIFEST_SCHEMA}, got {data.get('$schema')}"
        )

    tools = data.get("tools", [])
    logger.info(f"Loaded {len(tools)} tools from manifest: {path}")
    return tools


def update_manifest_cache(
    tool_defs: list[dict[str, Any]], manifest_path: Path | None = None
) -> None:
    """Update the manifest cache with tool definitions.

    This is used by the fetch_manifest.py script to update the manifest
    after fetching tools from the remote GitHub MCP server.

    Args:
        tool_defs: List of tool definitions from the remote server
        manifest_path: Path to save the manifest. Defaults to github_tools.json.
    """
    path = manifest_path or DEFAULT_MANIFEST_PATH

    # Preserve enabled status from existing manifest if present
    existing_enabled: dict[str, bool] = {}
    if path.exists():
        try:
            with open(path) as f:
                existing_data = json.load(f)
                for tool in existing_data.get("tools", []):
                    existing_enabled[tool["name"]] = tool.get("enabled", True)
        except Exception as e:
            logger.warning(f"Could not read existing manifest for enabled status: {e}")

    # Build tools list with enabled status
    tools_with_enabled = []
    for tool in tool_defs:
        tool_entry = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "inputSchema": tool.get("inputSchema", {}),
            # Preserve existing enabled status, default to True for new tools
            "enabled": existing_enabled.get(tool["name"], True),
        }
        tools_with_enabled.append(tool_entry)

    manifest = {
        "$schema": MANIFEST_SCHEMA,
        "metadata": {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "https://api.githubcopilot.com/mcp/",
            "tool_count": len(tools_with_enabled),
        },
        "tools": tools_with_enabled,
    }

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Updated manifest cache with {len(tools_with_enabled)} tools: {path}")


def _create_tool_function(tool_name: str) -> Any:
    """Create a tool function that proxies to the GitHub MCP server.

    This creates an async function that accepts keyword arguments
    and forwards them to the GitHub MCP server.

    Args:
        tool_name: Name of the tool to create

    Returns
    -------
        Async function that calls the GitHub MCP tool
    """

    async def tool_function(**kwargs: Any) -> Any:
        """Dynamically generated GitHub tool function."""
        return await call_github_tool(tool_name, kwargs)

    # Set function metadata
    tool_function.__name__ = f"github_{tool_name}"
    tool_function.__doc__ = f"Call the GitHub MCP '{tool_name}' tool."

    return tool_function


def _register_tools(tool_defs: list[dict[str, Any]]) -> int:
    """Register tool definitions with the MCP server.

    Args:
        tool_defs: List of tool definitions to register

    Returns
    -------
        Number of tools registered
    """
    registered_count = 0

    for tool_def in tool_defs:
        # Skip disabled tools
        if not tool_def.get("enabled", True):
            logger.debug(f"Skipping disabled tool: {tool_def['name']}")
            continue

        tool_name = tool_def["name"]
        description = tool_def.get("description", "")
        input_schema = tool_def.get("inputSchema", {})

        # Create the tool function
        tool_fn = _create_tool_function(tool_name)

        # Apply dr_mcp_extras for logging and tracing
        instrumented_fn = dr_mcp_extras()(tool_fn)

        # Ensure input_schema has the required structure
        if not input_schema or "type" not in input_schema:
            input_schema = {"type": "object", "properties": {}, "required": []}

        # Create FunctionTool directly with the inputSchema
        # This allows us to bypass function signature parsing and use the JSON schema directly
        tool = FunctionTool(
            fn=instrumented_fn,
            name=f"github_{tool_name}",
            description=description,
            parameters=input_schema,
            tags={"github"},
        )

        # Register with the MCP server
        mcp.add_tool(tool)
        registered_count += 1
        logger.debug(f"Registered GitHub tool: github_{tool_name}")

    logger.info(f"Registered {registered_count} GitHub tools")
    return registered_count


def register_github_tools(manifest_path: Path | None = None) -> int:
    """Register GitHub tools from the manifest.

    Loads tool definitions from the manifest file and registers them
    with the MCP server. Tools authenticate via OAuth at execution time.

    Args:
        manifest_path: Optional path to the manifest file

    Returns
    -------
        Number of tools registered
    """
    try:
        tool_defs = load_manifest(manifest_path)
    except FileNotFoundError:
        logger.error(
            "GitHub tools manifest not found. "
            "Run fetch_manifest.py to generate it, or ensure github_tools.json exists."
        )
        return 0
    except Exception as e:
        logger.error(f"Failed to load manifest: {e}")
        return 0

    return _register_tools(tool_defs)
