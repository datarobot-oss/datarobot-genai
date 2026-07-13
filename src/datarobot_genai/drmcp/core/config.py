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

import os
from typing import Any
from typing import Literal
from urllib.parse import urlparse
from urllib.parse import urlunparse

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from fastmcp.settings import DuplicateBehavior
from pydantic import Field
from pydantic import ValidationInfo
from pydantic import field_validator
from pydantic_settings import SettingsConfigDict

from datarobot_genai.drmcputils.constants import DEFAULT_DATAROBOT_ENDPOINT
from datarobot_genai.drmcputils.constants import RUNTIME_PARAM_ENV_VAR_NAME_PREFIX

from .constants import MCP_CLI_OPTS


class MCPToolConfig(DataRobotAppFrameworkBaseSettings):
    """Tool configuration for MCP server.

    Extends ``DataRobotAppFrameworkBaseSettings`` so each field resolves from env
    vars, ``.env``, file secrets, and ``pulumi_config.json``. Fields map by name:
    ``enable_predictive_tools`` reads ``ENABLE_PREDICTIVE_TOOLS`` (and the
    ``MLOPS_RUNTIME_PARAM_`` prefixed runtime-parameter variant), with payload
    extraction handled by the base settings sources.
    """

    enable_predictive_tools: bool = Field(
        default=False,
        description="Enable/disable predictive tools",
    )
    enable_jira_tools: bool = Field(
        default=False,
        description="Enable/disable Jira tools",
    )
    enable_confluence_tools: bool = Field(
        default=False,
        description="Enable/disable Confluence tools",
    )
    enable_gdrive_tools: bool = Field(
        default=False,
        description="Enable/disable GDrive tools",
    )
    enable_microsoft_graph_tools: bool = Field(
        default=False,
        description="Enable/disable Microsoft Graph (Sharepoint/OneDrive) tools",
    )
    enable_perplexity_tools: bool = Field(
        default=False,
        description="Enable/disable Perplexity tools",
    )
    enable_tavily_tools: bool = Field(
        default=False,
        description="Enable/disable Tavily search tools",
    )
    enable_dr_docs_tools: bool = Field(
        default=False,
        description="Enable/disable DataRobot documentation search tools",
    )
    enable_use_case_tools: bool = Field(
        default=False,
        description="Enable/disable use case tools",
    )
    enable_code_execution_tools: bool = Field(
        default=False,
        description="Enable/disable code execution tools",
    )
    enable_optimization_tools: bool = Field(
        default=False,
        description="Enable/disable optimization tools",
    )
    enable_vdb_tools: bool = Field(
        default=False,
        description="Enable/disable vector database tools",
    )
    enable_panels_tools: bool = Field(
        default=False,
        description="Enable/disable panel tools and resources",
    )
    enable_workload_tools: bool = Field(
        default=False,
        description="Enable/disable DataRobot Workload API tools",
    )
    enable_files_api_tools: bool = Field(
        default=False,
        description="Enable/disable DataRobot Files API (filesystem) tools",
    )

    # Treat empty env values as unset so a runtime parameter (resolved by the base
    # settings sources) is not shadowed by an empty plain env var.
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_ignore_empty=True,
    )


class MCPServerConfig(DataRobotAppFrameworkBaseSettings):
    """MCP Server configuration.

    Extends ``DataRobotAppFrameworkBaseSettings`` so each field resolves from env
    vars (including ``MLOPS_RUNTIME_PARAM_`` runtime parameters), ``.env``, file
    secrets, and ``pulumi_config.json``. Fields map by name: ``mcp_server_name``
    reads ``MCP_SERVER_NAME``.
    """

    mcp_server_name: str = Field(
        default="datarobot-mcp-server",
        description="Name of the MCP server",
    )
    mcp_server_port: int = Field(
        default=8080,
        description="Port number for the MCP server",
    )
    mcp_server_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="WARNING",
        description="Log level for the MCP server",
    )
    mcp_server_host: str = Field(
        default="0.0.0.0",
        description="Host address for the MCP server",
    )
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="App log level",
    )
    # When the server is run in a custom model, it is important to mount all routes under the
    # prefix provided in the URL_PREFIX
    mount_path: str = Field(default="/", alias="URL_PREFIX")

    @staticmethod
    def _get_default_otel_endpoint() -> str:
        """Get the default OpenTelemetry endpoint e.g. https://app.datarobot.com/otel."""
        parsed_url = urlparse(os.environ.get("DATAROBOT_ENDPOINT", DEFAULT_DATAROBOT_ENDPOINT))
        stripped_url = (parsed_url.scheme, parsed_url.netloc, "otel", "", "", "")
        return urlunparse(stripped_url)

    otel_collector_base_url: str = Field(
        default=_get_default_otel_endpoint(),
        description="Base URL for the OpenTelemetry collector",
    )
    otel_entity_id: str = Field(
        default="",
        description="Entity ID for tracing",
    )
    otel_attributes: dict[str, Any] = Field(
        default={},
        description="Attributes for tracing (as JSON string)",
    )
    otel_enabled: bool = Field(
        default=True,
        description="Enable/disable OpenTelemetry",
    )
    otel_enabled_http_instrumentors: bool = Field(
        default=False,
        description="Enable/disable HTTP instrumentors",
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="",
        description="Standard OTel OTLP endpoint. Takes priority over otel_collector_base_url.",
    )
    otel_exporter_otlp_headers: str = Field(
        default="",
        description="Standard OTel OTLP headers. Takes priority over entity_id construction.",
    )

    @field_validator("otel_exporter_otlp_headers", mode="before")
    @classmethod
    def _assemble_otel_headers(cls, v: object, info: ValidationInfo) -> object:
        if v:
            return v
        entity_id = (info.data or {}).get("otel_entity_id", "")
        api_token = os.environ.get("DATAROBOT_API_TOKEN", "")
        if entity_id and api_token:
            return f"x-datarobot-entity-id={entity_id},x-datarobot-api-key={api_token}"
        return v

    mcp_server_register_dynamic_tools_on_startup: bool = Field(
        default=False,
        description="Register dynamic tools on startup. When enabled, the MCP server will "
        "automatically register all DataRobot tool deployments as MCP tools during startup.",
    )
    mcp_server_enable_deployment_tool_provider: bool = Field(
        default=False,
        description="Serve tool-tagged deployments as MCP tools at request time. When "
        "enabled, the server re-discovers deployments tagged tool=tool on every "
        "tools/list, so newly tagged deployments appear without a restart (the "
        "zero-restart counterpart to mcp_server_register_dynamic_tools_on_startup).",
    )
    mcp_server_tool_registration_allow_empty_schema: bool = Field(
        default=False,
        description="Allow registration of tools with no input parameters. When enabled, "
        "tools can be registered with empty schemas for static endpoints that don't require any "
        "inputs. "
        "Disabled by default, as this is not typical use case and can hide potential issues with "
        "schema.",
    )
    mcp_server_tool_registration_duplicate_behavior: DuplicateBehavior = Field(
        default="warn",
        description="Behavior when a tool with the same name already exists in the MCP server. "
        " - 'warn': will log a warning and replace the existing tool. "
        " - 'replace': will replace the existing tool without a warning. "
        " - 'error': will raise an error and prevent registration. "
        " - 'ignore': will skip registration of the new tool.",
    )
    mcp_server_register_dynamic_prompts_on_startup: bool = Field(
        default=False,
        description="Register dynamic prompts on startup. When enabled, the MCP server will "
        "automatically register all prompts from DataRobot Prompt Management "
        "as MCP prompts during startup.",
    )
    mcp_server_prompt_registration_duplicate_behavior: DuplicateBehavior = Field(
        default="warn",
        description="Behavior when a prompt with the same name already exists in the MCP server. "
        " - 'warn': will log a warning and replace the existing tool. "
        " - 'replace': will replace the existing tool without a warning. "
        " - 'error': will raise an error and prevent registration. "
        " - 'ignore': will skip registration of the new tool.",
    )
    mcp_cli_configs: str | None = Field(
        default=None,
        description="Comma-separated list of features to enable: dynamic_tools, dynamic_prompts, "
        "predictive, gdrive, microsoft_graph, jira, confluence, perplexity, tavily. "
        "When unset (None), defaults apply. When set to empty string, all listed features are "
        "disabled. Individual env vars (e.g. ENABLE_PREDICTIVE_TOOLS) take precedence when set.",
    )

    tool_config: MCPToolConfig = Field(
        default_factory=MCPToolConfig,
        description="Tool configuration",
    )

    # Treat empty env values as unset so a runtime parameter (resolved by the base
    # settings sources) is not shadowed by an empty plain env var.
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_ignore_empty=True,
    )


# Global configuration instance
_config: MCPServerConfig | None = None


def _individual_env_set_for_field(attr: str) -> bool:
    """Return True if the user set this field via env.

    ``DataRobotAppFrameworkBaseSettings`` resolves fields by name, so a field maps
    to its uppercase env var (e.g. ``ENABLE_GDRIVE_TOOLS``) and the DataRobot
    runtime-parameter variant (``MLOPS_RUNTIME_PARAM_ENABLE_GDRIVE_TOOLS``).
    """
    env_name = attr.upper()
    names = [env_name, RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + env_name]
    return any(os.environ.get(name, "").strip() for name in names)


def _apply_mcp_cli_configs_overrides(config: MCPServerConfig) -> MCPServerConfig:
    """Apply MCP_CLI_CONFIGS from config: listed options enabled; others disabled
    unless individual env set.

    - If mcp_cli_configs is None (unset), do not apply overrides (defaults apply).
    - If mcp_cli_configs is set but empty string, treat as disable all (enabled set is empty).
    - If mcp_cli_configs is set and non-empty, parse comma list and apply as before.
    """
    if config.mcp_cli_configs is None:
        return config
    raw = (config.mcp_cli_configs or "").strip()
    enabled = {s.strip().lower() for s in raw.split(",") if s.strip()} if raw else set()
    root_updates: dict[str, Any] = {}
    tool_updates: dict[str, Any] = {}
    for mcp_opt, root_attr, tool_attr in MCP_CLI_OPTS:
        attr = root_attr if root_attr is not None else tool_attr
        assert attr is not None  # each MCP_CLI_OPTS row has either root_attr or tool_attr
        if _individual_env_set_for_field(attr):
            continue
        if mcp_opt in enabled:
            if root_attr is not None:
                current = getattr(config, root_attr)
                default = MCPServerConfig.model_fields[root_attr].default
                if current == default:
                    root_updates[root_attr] = True
            else:
                assert tool_attr is not None
                current = getattr(config.tool_config, tool_attr)
                default = MCPToolConfig.model_fields[tool_attr].default
                if current == default:
                    tool_updates[tool_attr] = True
        elif root_attr is not None:
            # Option not in MCP_CLI_CONFIGS: disable.
            root_updates[root_attr] = False
        else:
            assert tool_attr is not None
            tool_updates[tool_attr] = False
    if tool_updates:
        root_updates["tool_config"] = config.tool_config.model_copy(update=tool_updates)
    if not root_updates:
        return config
    return config.model_copy(update=root_updates)


def get_config() -> MCPServerConfig:
    """Get the global configuration instance."""
    # Use a local variable to avoid global statement warning
    config = _config
    if config is None:
        config = MCPServerConfig()
        config = _apply_mcp_cli_configs_overrides(config)
        # Update the global variable
        globals()["_config"] = config
    return config
