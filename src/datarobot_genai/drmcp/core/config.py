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

from fastmcp.settings import DuplicateBehavior
from pydantic import AliasChoices
from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from .config_utils import extract_datarobot_dict_runtime_param_payload
from .config_utils import extract_datarobot_runtime_param_payload
from .constants import DEFAULT_DATAROBOT_ENDPOINT
from .constants import MCP_CLI_OPTS
from .constants import RUNTIME_PARAM_ENV_VAR_NAME_PREFIX


class MCPToolConfig(BaseSettings):
    """Tool configuration for MCP server."""

    enable_predictive_tools: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "ENABLE_PREDICTIVE_TOOLS",
            "ENABLE_PREDICTIVE_TOOLS",
        ),
        description="Enable/disable predictive tools",
    )

    enable_jira_tools: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "ENABLE_JIRA_TOOLS",
            "ENABLE_JIRA_TOOLS",
        ),
        description="Enable/disable Jira tools",
    )

    enable_confluence_tools: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "ENABLE_CONFLUENCE_TOOLS",
            "ENABLE_CONFLUENCE_TOOLS",
        ),
        description="Enable/disable Confluence tools",
    )

    enable_gdrive_tools: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "ENABLE_GDRIVE_TOOLS",
            "ENABLE_GDRIVE_TOOLS",
        ),
        description="Enable/disable GDrive tools",
    )

    enable_microsoft_graph_tools: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "ENABLE_MICROSOFT_GRAPH_TOOLS",
            "ENABLE_MICROSOFT_GRAPH_TOOLS",
        ),
        description="Enable/disable Microsoft Graph (Sharepoint/OneDrive) tools",
    )

    enable_perplexity_tools: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "ENABLE_PERPLEXITY_TOOLS",
            "ENABLE_PERPLEXITY_TOOLS",
        ),
        description="Enable/disable Perplexity tools",
    )

    enable_tavily_tools: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "ENABLE_TAVILY_TOOLS",
            "ENABLE_TAVILY_TOOLS",
        ),
        description="Enable/disable Tavily search tools",
    )

    enable_dr_docs_tools: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "ENABLE_DR_DOCS_TOOLS",
            "ENABLE_DR_DOCS_TOOLS",
        ),
        description="Enable/disable DataRobot documentation search tools",
    )

    is_atlassian_oauth_provider_configured: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "IS_ATLASSIAN_OAUTH_PROVIDER_CONFIGURED",
            "IS_ATLASSIAN_OAUTH_PROVIDER_CONFIGURED",
        ),
        description="Whether Atlassian OAuth provider is configured for Atlassian integration",
    )

    @property
    def is_atlassian_oauth_configured(self) -> bool:
        """Check if Atlassian OAuth is configured via provider flag or environment variables."""
        return self.is_atlassian_oauth_provider_configured or bool(
            os.getenv("ATLASSIAN_CLIENT_ID") and os.getenv("ATLASSIAN_CLIENT_SECRET")
        )

    is_google_oauth_provider_configured: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "IS_GOOGLE_OAUTH_PROVIDER_CONFIGURED",
            "IS_GOOGLE_OAUTH_PROVIDER_CONFIGURED",
        ),
        description="Whether Google OAuth provider is configured for Google integration",
    )

    @property
    def is_google_oauth_configured(self) -> bool:
        return self.is_google_oauth_provider_configured or bool(
            os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET")
        )

    is_microsoft_oauth_provider_configured: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "IS_MICROSOFT_OAUTH_PROVIDER_CONFIGURED",
            "IS_MICROSOFT_OAUTH_PROVIDER_CONFIGURED",
        ),
        description="Whether Microsoft OAuth provider is configured for Microsoft integration",
    )

    @property
    def is_microsoft_oauth_configured(self) -> bool:
        return self.is_microsoft_oauth_provider_configured or bool(
            os.getenv("MICROSOFT_CLIENT_ID") and os.getenv("MICROSOFT_CLIENT_SECRET")
        )

    @field_validator(
        "enable_predictive_tools",
        "enable_jira_tools",
        "enable_confluence_tools",
        "enable_gdrive_tools",
        "enable_microsoft_graph_tools",
        "enable_perplexity_tools",
        "enable_tavily_tools",
        "enable_dr_docs_tools",
        "is_atlassian_oauth_provider_configured",
        "is_google_oauth_provider_configured",
        "is_microsoft_oauth_provider_configured",
        mode="before",
    )
    @classmethod
    def validate_runtime_params(cls, v: Any) -> Any:
        """Validate runtime parameters."""
        return extract_datarobot_runtime_param_payload(v)

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_file_encoding="utf-8",
        extra="ignore",
    )


class MCPServerConfig(BaseSettings):
    """MCP Server configuration using pydantic settings."""

    mcp_server_name: str = Field(
        default="datarobot-mcp-server",
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "MCP_SERVER_NAME",
            "MCP_SERVER_NAME",
        ),
        description="Name of the MCP server",
    )
    mcp_server_port: int = Field(
        default=8080,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "MCP_SERVER_PORT",
            "MCP_SERVER_PORT",
        ),
        description="Port number for the MCP server",
    )
    mcp_server_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="WARNING",
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "MCP_SERVER_LOG_LEVEL",
            "MCP_SERVER_LOG_LEVEL",
        ),
        description="Log level for the MCP server",
    )
    mcp_server_host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "MCP_SERVER_HOST",
            "MCP_SERVER_HOST",
        ),
        description="Host address for the MCP server",
    )
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "APP_LOG_LEVEL",
            "APP_LOG_LEVEL",
        ),
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
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "OTEL_COLLECTOR_BASE_URL",
            "OTEL_COLLECTOR_BASE_URL",
        ),
        description="Base URL for the OpenTelemetry collector",
    )
    otel_entity_id: str = Field(
        default="",
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "OTEL_ENTITY_ID",
            "OTEL_ENTITY_ID",
        ),
        description="Entity ID for tracing",
    )
    otel_attributes: dict[str, Any] = Field(
        default={},
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "OTEL_ATTRIBUTES",
            "OTEL_ATTRIBUTES",
        ),
        description="Attributes for tracing (as JSON string)",
    )
    otel_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "OTEL_ENABLED",
            "OTEL_ENABLED",
        ),
        description="Enable/disable OpenTelemetry",
    )
    otel_enabled_http_instrumentors: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "OTEL_ENABLED_HTTP_INSTRUMENTORS",
            "OTEL_ENABLED_HTTP_INSTRUMENTORS",
        ),
        description="Enable/disable HTTP instrumentors",
    )
    mcp_server_register_dynamic_tools_on_startup: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "MCP_SERVER_REGISTER_DYNAMIC_TOOLS_ON_STARTUP",
            "MCP_SERVER_REGISTER_DYNAMIC_TOOLS_ON_STARTUP",
        ),
        description="Register dynamic tools on startup. When enabled, the MCP server will "
        "automatically register all DataRobot tool deployments as MCP tools during startup.",
    )
    tool_registration_allow_empty_schema: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "MCP_SERVER_TOOL_REGISTRATION_ALLOW_EMPTY_SCHEMA",
            "MCP_SERVER_TOOL_REGISTRATION_ALLOW_EMPTY_SCHEMA",
        ),
        description="Allow registration of tools with no input parameters. When enabled, "
        "tools can be registered with empty schemas for static endpoints that don't require any "
        "inputs. "
        "Disabled by default, as this is not typical use case and can hide potential issues with "
        "schema.",
    )
    tool_registration_duplicate_behavior: DuplicateBehavior = Field(
        default="warn",
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "MCP_SERVER_TOOL_REGISTRATION_DUPLICATE_BEHAVIOR",
            "MCP_SERVER_TOOL_REGISTRATION_DUPLICATE_BEHAVIOR",
        ),
        description="Behavior when a tool with the same name already exists in the MCP server. "
        " - 'warn': will log a warning and replace the existing tool. "
        " - 'replace': will replace the existing tool without a warning. "
        " - 'error': will raise an error and prevent registration. "
        " - 'ignore': will skip registration of the new tool.",
    )
    mcp_server_register_dynamic_prompts_on_startup: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "MCP_SERVER_REGISTER_DYNAMIC_PROMPTS_ON_STARTUP",
            "MCP_SERVER_REGISTER_DYNAMIC_PROMPTS_ON_STARTUP",
        ),
        description="Register dynamic prompts on startup. When enabled, the MCP server will "
        "automatically register all prompts from DataRobot Prompt Management "
        "as MCP prompts during startup.",
    )
    prompt_registration_duplicate_behavior: DuplicateBehavior = Field(
        default="warn",
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "MCP_SERVER_PROMPT_REGISTRATION_DUPLICATE_BEHAVIOR",
            "MCP_SERVER_PROMPT_REGISTRATION_DUPLICATE_BEHAVIOR",
        ),
        description="Behavior when a prompt with the same name already exists in the MCP server. "
        " - 'warn': will log a warning and replace the existing tool. "
        " - 'replace': will replace the existing tool without a warning. "
        " - 'error': will raise an error and prevent registration. "
        " - 'ignore': will skip registration of the new tool.",
    )
    enable_memory_management: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "ENABLE_MEMORY_MANAGEMENT",
            "ENABLE_MEMORY_MANAGEMENT",
        ),
        description="Enable/disable memory management",
    )
    mcp_cli_configs: str = Field(
        default="",
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "MCP_CLI_CONFIGS",
            "MCP_CLI_CONFIGS",
        ),
        description="Comma-separated list of features to enable: dynamic_tools, dynamic_prompts, "
        "predictive, gdrive, microsoft_graph, jira, confluence, perplexity, tavily. "
        "Individual env vars (e.g. ENABLE_PREDICTIVE_TOOLS) take precedence when set.",
    )

    tool_config: MCPToolConfig = Field(
        default_factory=MCPToolConfig,
        description="Tool configuration",
    )

    @field_validator(
        "otel_attributes",
        mode="before",
    )
    @classmethod
    def validate_dict_runtime_params(cls, v: Any) -> Any:
        """Validate dict runtime parameters."""
        return extract_datarobot_dict_runtime_param_payload(v)

    @field_validator(
        "mcp_server_name",
        "mcp_server_log_level",
        "app_log_level",
        "otel_collector_base_url",
        "otel_entity_id",
        "otel_enabled",
        "otel_enabled_http_instrumentors",
        "enable_memory_management",
        "mcp_cli_configs",
        "tool_registration_allow_empty_schema",
        "mcp_server_register_dynamic_tools_on_startup",
        "tool_registration_duplicate_behavior",
        "mcp_server_register_dynamic_prompts_on_startup",
        mode="before",
    )
    @classmethod
    def validate_runtime_params(cls, v: Any) -> Any:
        """Validate runtime parameters."""
        return extract_datarobot_runtime_param_payload(v)

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Global configuration instance
_config: MCPServerConfig | None = None


def _individual_env_set_for_field(
    model: type[MCPServerConfig] | type[MCPToolConfig], attr: str
) -> bool:
    """Return True if the user set this field via env (any of its validation_alias names)."""
    info = model.model_fields[attr]
    alias = getattr(info, "validation_alias", None)
    if alias is None:
        return False
    if isinstance(alias, str):
        names = [alias]
    elif hasattr(alias, "choices"):
        names = list(alias.choices)
    else:
        names = []
    return any(os.environ.get(name, "").strip() for name in names)


def _apply_mcp_cli_configs_overrides(config: MCPServerConfig) -> MCPServerConfig:
    """Override from MCP_CLI_CONFIGS when field is at default and individual env not set."""
    raw = (config.mcp_cli_configs or "").strip()
    if not raw:
        return config
    enabled = {s.strip().lower() for s in raw.split(",") if s.strip()}
    root_updates: dict[str, Any] = {}
    tool_updates: dict[str, Any] = {}
    for mcp_opt, root_attr, tool_attr in MCP_CLI_OPTS:
        if mcp_opt not in enabled:
            continue
        attr = root_attr if root_attr is not None else tool_attr
        assert attr is not None  # each MCP_CLI_OPTS row has either root_attr or tool_attr
        model: type[MCPServerConfig] | type[MCPToolConfig] = (
            MCPServerConfig if root_attr else MCPToolConfig
        )
        if _individual_env_set_for_field(model, attr):
            continue
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
