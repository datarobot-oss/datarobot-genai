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

"""Tool configuration and enablement logic."""

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING
from typing import TypedDict

if TYPE_CHECKING:
    from .config import MCPServerConfig


class ToolType(str, Enum):
    """Enumeration of available tool types."""

    PREDICTIVE = "predictive"
    JIRA = "jira"
    CONFLUENCE = "confluence"
    GDRIVE = "gdrive"
    MICROSOFT_GRAPH = "microsoft_graph"
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"


class ToolConfig(TypedDict):
    """Configuration for a tool type."""

    name: str
    oauth_check: Callable[["MCPServerConfig"], bool] | None
    directory: str
    package_prefix: str
    config_field_name: str


# Tool configuration registry
TOOL_CONFIGS: dict[ToolType, ToolConfig] = {
    ToolType.PREDICTIVE: ToolConfig(
        name="predictive",
        oauth_check=None,
        directory="predictive",
        package_prefix="datarobot_genai.drmcp.tools.predictive",
        config_field_name="enable_predictive_tools",
    ),
    ToolType.JIRA: ToolConfig(
        name="jira",
        oauth_check=lambda config: config.tool_config.is_atlassian_oauth_configured,
        directory="jira",
        package_prefix="datarobot_genai.drmcp.tools.jira",
        config_field_name="enable_jira_tools",
    ),
    ToolType.CONFLUENCE: ToolConfig(
        name="confluence",
        oauth_check=lambda config: config.tool_config.is_atlassian_oauth_configured,
        directory="confluence",
        package_prefix="datarobot_genai.drmcp.tools.confluence",
        config_field_name="enable_confluence_tools",
    ),
    ToolType.GDRIVE: ToolConfig(
        name="gdrive",
        oauth_check=lambda config: config.tool_config.is_google_oauth_configured,
        directory="gdrive",
        package_prefix="datarobot_genai.drmcp.tools.gdrive",
        config_field_name="enable_gdrive_tools",
    ),
    ToolType.MICROSOFT_GRAPH: ToolConfig(
        name="microsoft_graph",
        oauth_check=lambda config: config.tool_config.is_microsoft_oauth_configured,
        directory="microsoft_graph",
        package_prefix="datarobot_genai.drmcp.tools.microsoft_graph",
        config_field_name="enable_microsoft_graph_tools",
    ),
    ToolType.PERPLEXITY: ToolConfig(
        name="perplexity",
        oauth_check=None,  # OAuth for Perplexity is not supported
        directory="perplexity",
        package_prefix="datarobot_genai.drmcp.tools.perplexity",
        config_field_name="enable_perplexity_tools",
    ),
    ToolType.TAVILY: ToolConfig(
        name="tavily",
        oauth_check=None,
        directory="tavily",
        package_prefix="datarobot_genai.drmcp.tools.tavily",
        config_field_name="enable_tavily_tools",
    ),
}


def get_tool_enable_config_name(tool_type: ToolType) -> str:
    """Get the configuration field name for enabling a tool."""
    return TOOL_CONFIGS[tool_type]["config_field_name"]


def is_tool_enabled(tool_type: ToolType, config: "MCPServerConfig") -> bool:
    """
    Check if a tool is enabled based on configuration.

    Args:
        tool_type: The type of tool to check
        config: The server configuration

    Returns
    -------
        True if the tool is enabled, False otherwise
    """
    tool_config_registry = TOOL_CONFIGS[tool_type]
    enable_config_name = tool_config_registry["config_field_name"]
    is_enabled = getattr(config.tool_config, enable_config_name)

    # If tool is enabled, check OAuth requirements if needed
    if is_enabled and tool_config_registry["oauth_check"] is not None:
        return tool_config_registry["oauth_check"](config)

    return is_enabled
