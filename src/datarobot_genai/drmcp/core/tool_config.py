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

from enum import StrEnum
from typing import TYPE_CHECKING
from typing import TypedDict

if TYPE_CHECKING:
    from .config import MCPServerConfig


class ToolType(StrEnum):
    """Enumeration of available tool types."""

    PREDICTIVE = "predictive"
    JIRA = "jira"
    CONFLUENCE = "confluence"
    GDRIVE = "gdrive"
    MICROSOFT_GRAPH = "microsoft_graph"
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DR_DOCS = "dr_docs"
    USE_CASE = "use_case"
    CODE_EXECUTION = "code_execution"
    OPTIMIZATION = "optimization"
    VDB = "vdb"


class ToolConfig(TypedDict):
    """Configuration for a tool type."""

    name: str
    directory: str
    package_prefix: str
    config_field_name: str


# Tool configuration registry
TOOL_CONFIGS: dict[ToolType, ToolConfig] = {
    ToolType.PREDICTIVE: ToolConfig(
        name="predictive",
        directory="predictive",
        package_prefix="datarobot_genai.drtools.predictive",
        config_field_name="enable_predictive_tools",
    ),
    ToolType.JIRA: ToolConfig(
        name="jira",
        directory="jira",
        package_prefix="datarobot_genai.drtools.jira",
        config_field_name="enable_jira_tools",
    ),
    ToolType.CONFLUENCE: ToolConfig(
        name="confluence",
        directory="confluence",
        package_prefix="datarobot_genai.drtools.confluence",
        config_field_name="enable_confluence_tools",
    ),
    ToolType.GDRIVE: ToolConfig(
        name="gdrive",
        directory="gdrive",
        package_prefix="datarobot_genai.drtools.gdrive",
        config_field_name="enable_gdrive_tools",
    ),
    ToolType.MICROSOFT_GRAPH: ToolConfig(
        name="microsoft_graph",
        directory="microsoft_graph",
        package_prefix="datarobot_genai.drtools.microsoft_graph",
        config_field_name="enable_microsoft_graph_tools",
    ),
    ToolType.PERPLEXITY: ToolConfig(
        name="perplexity",
        directory="perplexity",
        package_prefix="datarobot_genai.drtools.perplexity",
        config_field_name="enable_perplexity_tools",
    ),
    ToolType.TAVILY: ToolConfig(
        name="tavily",
        directory="tavily",
        package_prefix="datarobot_genai.drtools.tavily",
        config_field_name="enable_tavily_tools",
    ),
    ToolType.DR_DOCS: ToolConfig(
        name="dr_docs",
        directory="dr_docs",
        package_prefix="datarobot_genai.drtools.dr_docs",
        config_field_name="enable_dr_docs_tools",
    ),
    ToolType.USE_CASE: ToolConfig(
        name="use_case",
        directory="use_case",
        package_prefix="datarobot_genai.drtools.use_case",
        config_field_name="enable_use_case_tools",
    ),
    ToolType.CODE_EXECUTION: ToolConfig(
        name="code_execution",
        directory="code_execution",
        package_prefix="datarobot_genai.drtools.code_execution",
        config_field_name="enable_code_execution_tools",
    ),
    ToolType.OPTIMIZATION: ToolConfig(
        name="optimization",
        directory="optimization",
        package_prefix="datarobot_genai.drtools.optimization",
        config_field_name="enable_optimization_tools",
    ),
    ToolType.VDB: ToolConfig(
        name="vdb",
        directory="vdb",
        package_prefix="datarobot_genai.drtools.vdb",
        config_field_name="enable_vdb_tools",
    ),
}


def get_tool_enable_config_name(tool_type: ToolType) -> str:
    """Get the configuration field name for enabling a tool."""
    return TOOL_CONFIGS[tool_type]["config_field_name"]


def is_tool_enabled(tool_type: ToolType, config: "MCPServerConfig") -> bool:
    """Return whether *tool_type* is enabled in *config*."""
    enable_config_name = TOOL_CONFIGS[tool_type]["config_field_name"]
    return bool(getattr(config.tool_config, enable_config_name))
