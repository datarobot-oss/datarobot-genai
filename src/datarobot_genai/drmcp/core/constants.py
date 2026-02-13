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


MAX_INLINE_SIZE = 1024 * 1024  # 1MB
DEFAULT_DATAROBOT_ENDPOINT = "https://app.datarobot.com/api/v2"
RUNTIME_PARAM_ENV_VAR_NAME_PREFIX = "MLOPS_RUNTIME_PARAM_"

# Streamable-HTTP MCP endpoint path (used with prefix_mount_path for full path).
MCP_PATH_ENDPOINT = "/mcp"

# mcp_opt, root_attr, tool_attr. Used by config to apply MCP_CLI_CONFIGS overrides.
MCP_CLI_OPTS: list[tuple[str, str | None, str | None]] = [
    ("dynamic_tools", "mcp_server_register_dynamic_tools_on_startup", None),
    ("dynamic_prompts", "mcp_server_register_dynamic_prompts_on_startup", None),
    ("predictive", None, "enable_predictive_tools"),
    ("gdrive", None, "enable_gdrive_tools"),
    ("microsoft_graph", None, "enable_microsoft_graph_tools"),
    ("jira", None, "enable_jira_tools"),
    ("confluence", None, "enable_confluence_tools"),
    ("perplexity", None, "enable_perplexity_tools"),
    ("tavily", None, "enable_tavily_tools"),
]
