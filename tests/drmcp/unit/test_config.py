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
from typing import get_args
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core import config as config_module
from datarobot_genai.drmcp.core.config import MCPServerConfig
from datarobot_genai.drmcp.core.config import get_config
from datarobot_genai.drmcp.core.mcp_instance import DataRobotMCP


def test_config_defaults() -> None:
    """Test that the important default configuration values are as expected."""
    # Clear any environment variables that might override defaults
    with patch.dict(os.environ, clear=True):
        # Clear the cached config instance to ensure we get a fresh one
        config_module._config = None

        # Create a new config instance without loading from .env file
        config = MCPServerConfig(_env_file=None)

        # Dynamic tools registration should be disabled by default
        # as it can cause startup delays and is not always desired.
        assert config.mcp_server_register_dynamic_tools_on_startup is False

        # The default behavior for duplicate tool registrations is aligned
        # with FastMCP default.
        assert config.tool_registration_duplicate_behavior == "warn"

        # Dynamic prompt registration should be disabled by default
        # as it can cause startup delays and is not always desired.
        assert config.mcp_server_register_dynamic_prompts_on_startup is False

        # The default behavior for duplicate prompt registrations is aligned
        # with FastMCP default.
        assert config.prompt_registration_duplicate_behavior == "warn"

        # Tool enablement defaults
        assert config.tool_config.enable_predictive_tools is True
        assert config.tool_config.enable_jira_tools is False
        assert config.tool_config.enable_confluence_tools is False
        assert config.tool_config.enable_microsoft_graph_tools is False
        assert config.tool_config.enable_perplexity_tools is False

        # OAuth provider configuration defaults
        assert config.tool_config.is_atlassian_oauth_provider_configured is False
        assert config.tool_config.is_microsoft_oauth_provider_configured is False

        # Clean up the cached config after the test
        config_module._config = None


class TestDuplicateBehavior:
    def test_allowed_duplicate_behaviors(self) -> None:
        """Test that the allowed duplicate behaviors are as expected."""
        expected_behaviors = {"error", "warn", "ignore", "replace"}

        # Get the type annotation from the model field to check if the
        # allowed values match, if there are any changes in the future,
        # please review the tool registration logic to ensure it still works
        # as intended.
        tools_field_info = MCPServerConfig.model_fields["tool_registration_duplicate_behavior"]
        prompts_field_info = MCPServerConfig.model_fields["prompt_registration_duplicate_behavior"]
        allowed_tools_behaviors = set(get_args(tools_field_info.annotation))
        allowed_prompts_behaviors = set(get_args(prompts_field_info.annotation))

        assert expected_behaviors == allowed_tools_behaviors
        assert expected_behaviors == allowed_prompts_behaviors

    @pytest.mark.parametrize("value", ["error", "warn", "replace", "ignore"])
    def test_setting_is_propagated_correctly(self, value) -> None:
        """Test that setting the duplicate behavior is propagated correctly to the tool manager."""
        tools_env_var = "MCP_SERVER_TOOL_REGISTRATION_DUPLICATE_BEHAVIOR"
        prompts_env_var = "MCP_SERVER_PROMPT_REGISTRATION_DUPLICATE_BEHAVIOR"

        with patch.dict(os.environ, {tools_env_var: value, prompts_env_var: value}, clear=False):
            # Create a fresh config instance that reads from the updated
            # environment without affecting the global mcp instance
            # shared across tests.
            config = MCPServerConfig()

            test_mcp = DataRobotMCP(
                name=config.mcp_server_name,
                on_duplicate_tools=config.tool_registration_duplicate_behavior,
                on_duplicate_prompts=config.prompt_registration_duplicate_behavior,
            )

            # Verify the setting was applied correctly
            assert test_mcp._tool_manager.duplicate_behavior == value
            assert test_mcp._prompt_manager.duplicate_behavior == value


class TestToolConfiguration:
    """Test tool enablement and OAuth configuration."""

    def test_tool_enablement_defaults(self) -> None:
        """Test that tool enablement defaults are correct."""
        with patch.dict(os.environ, clear=True):
            config_module._config = None
            config = MCPServerConfig(_env_file=None)

            assert config.tool_config.enable_predictive_tools is True
            assert config.tool_config.enable_jira_tools is False
            assert config.tool_config.enable_confluence_tools is False
            assert config.tool_config.enable_microsoft_graph_tools is False
            assert config.tool_config.enable_perplexity_tools is False

            config_module._config = None

    @pytest.mark.parametrize(
        "tool_name,env_var",
        [
            ("enable_predictive_tools", "ENABLE_PREDICTIVE_TOOLS"),
            ("enable_jira_tools", "ENABLE_JIRA_TOOLS"),
            ("enable_confluence_tools", "ENABLE_CONFLUENCE_TOOLS"),
            ("enable_gdrive_tools", "ENABLE_GDRIVE_TOOLS"),
            ("enable_microsoft_graph_tools", "ENABLE_MICROSOFT_GRAPH_TOOLS"),
            ("enable_perplexity_tools", "ENABLE_PERPLEXITY_TOOLS"),
        ],
    )
    def test_tool_enablement_via_env_var(self, tool_name: str, env_var: str) -> None:
        """Test that tool enablement can be set via environment variables."""
        with patch.dict(os.environ, {env_var: "true"}, clear=False):
            config = MCPServerConfig()
            assert getattr(config.tool_config, tool_name) is True

        with patch.dict(os.environ, {env_var: "false"}, clear=False):
            config = MCPServerConfig()
            assert getattr(config.tool_config, tool_name) is False

    def test_atlassian_oauth_configured_via_provider_flag(self) -> None:
        """Test is_atlassian_oauth_configured when provider flag is set."""
        with patch.dict(
            os.environ, {"IS_ATLASSIAN_OAUTH_PROVIDER_CONFIGURED": "true"}, clear=False
        ):
            config = MCPServerConfig()
            assert config.tool_config.is_atlassian_oauth_configured is True

    def test_atlassian_oauth_configured_via_env_vars(self) -> None:
        """Test is_atlassian_oauth_configured when env vars are set."""
        with patch.dict(
            os.environ,
            {"ATLASSIAN_CLIENT_ID": "test_id", "ATLASSIAN_CLIENT_SECRET": "test_secret"},
            clear=False,
        ):
            config = MCPServerConfig()
            assert config.tool_config.is_atlassian_oauth_configured is True

    def test_atlassian_oauth_not_configured(self) -> None:
        """Test is_atlassian_oauth_configured when not configured."""
        with patch.dict(os.environ, clear=True):
            config_module._config = None
            config = MCPServerConfig(_env_file=None)
            assert config.tool_config.is_atlassian_oauth_configured is False
            config_module._config = None

    def test_atlassian_oauth_partial_env_vars(self) -> None:
        """Test is_atlassian_oauth_configured with only one env var set."""
        with patch.dict(os.environ, {"ATLASSIAN_CLIENT_ID": "test_id"}, clear=False):
            config = MCPServerConfig()
            assert config.tool_config.is_atlassian_oauth_configured is False

    def test_gdrive_oauth_configured_via_provider_flag(self) -> None:
        """Test is_google_oauth_configured when provider flag is set."""
        with patch.dict(os.environ, {"IS_GOOGLE_OAUTH_PROVIDER_CONFIGURED": "true"}, clear=False):
            config = MCPServerConfig()
            assert config.tool_config.is_google_oauth_configured is True

    def test_gdrive_oauth_configured_via_env_vars(self) -> None:
        """Test is_google_oauth_configured when env vars are set."""
        with patch.dict(
            os.environ,
            {"GOOGLE_CLIENT_ID": "test_id", "GOOGLE_CLIENT_SECRET": "test_secret"},
            clear=False,
        ):
            config = MCPServerConfig()
            assert config.tool_config.is_google_oauth_configured is True

    def test_gdrive_oauth_not_configured(self) -> None:
        """Test is_google_oauth_configured when not configured."""
        with patch.dict(os.environ, clear=True):
            config_module._config = None
            config = MCPServerConfig(_env_file=None)
            assert config.tool_config.is_google_oauth_configured is False
            config_module._config = None

    def test_gdrive_oauth_partial_env_vars(self) -> None:
        """Test is_google_oauth_configured with only one env var set."""
        with patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "test_id"}, clear=False):
            config = MCPServerConfig()
            assert config.tool_config.is_google_oauth_configured is False

    def test_microsoft_oauth_configured_via_provider_flag(self) -> None:
        """Test is_microsoft_oauth_configured when provider flag is set."""
        with patch.dict(
            os.environ, {"IS_MICROSOFT_OAUTH_PROVIDER_CONFIGURED": "true"}, clear=False
        ):
            config = MCPServerConfig()
            assert config.tool_config.is_microsoft_oauth_configured is True

    def test_microsoft_oauth_configured_via_env_vars(self) -> None:
        """Test is_microsoft_oauth_configured when env vars are set."""
        with patch.dict(
            os.environ,
            {"MICROSOFT_CLIENT_ID": "test_id", "MICROSOFT_CLIENT_SECRET": "test_secret"},
            clear=False,
        ):
            config = MCPServerConfig()
            assert config.tool_config.is_microsoft_oauth_configured is True

    def test_microsoft_oauth_not_configured(self) -> None:
        """Test is_microsoft_oauth_configured when not configured."""
        with patch.dict(os.environ, clear=True):
            config_module._config = None
            config = MCPServerConfig(_env_file=None)
            assert config.tool_config.is_microsoft_oauth_configured is False
            config_module._config = None

    def test_microsoft_oauth_partial_env_vars(self) -> None:
        """Test is_microsoft_oauth_configured with only one env var set."""
        with patch.dict(os.environ, {"MICROSOFT_CLIENT_ID": "test_id"}, clear=False):
            config = MCPServerConfig()
            assert config.tool_config.is_microsoft_oauth_configured is False


class TestMCPCLIConfigs:
    """Test MCP_CLI_CONFIGS override: only when option in list, at default, and env not set."""

    def _reset_config(self) -> None:
        config_module._config = None

    def test_no_mcp_cli_configs_unset(self) -> None:
        """When MCP_CLI_CONFIGS is unset, no overrides; all fields keep model defaults."""
        with patch.dict(os.environ, {}, clear=True):
            self._reset_config()
            config = get_config()
            assert config.mcp_cli_configs == ""
            assert config.mcp_server_register_dynamic_tools_on_startup is False
            assert config.mcp_server_register_dynamic_prompts_on_startup is False
            assert config.tool_config.enable_predictive_tools is True
            assert config.tool_config.enable_gdrive_tools is False
            assert config.tool_config.enable_jira_tools is False
            assert config.tool_config.enable_confluence_tools is False
            assert config.tool_config.enable_microsoft_graph_tools is False
            assert config.tool_config.enable_perplexity_tools is False
            assert config.tool_config.enable_tavily_tools is False
            self._reset_config()

    def test_no_mcp_cli_configs_empty(self) -> None:
        """When MCP_CLI_CONFIGS is empty string, no overrides."""
        with patch.dict(os.environ, {"MCP_CLI_CONFIGS": ""}, clear=True):
            self._reset_config()
            config = get_config()
            assert config.mcp_server_register_dynamic_tools_on_startup is False
            assert config.tool_config.enable_gdrive_tools is False
            assert config.tool_config.enable_jira_tools is False
            assert config.tool_config.enable_confluence_tools is False
            assert config.tool_config.enable_microsoft_graph_tools is False
            assert config.tool_config.enable_perplexity_tools is False
            assert config.tool_config.enable_tavily_tools is False
            self._reset_config()

    def test_mcp_cli_configs_single_option_enables_root_field(self) -> None:
        """Option in MCP_CLI_CONFIGS with no individual env overrides root field to True."""
        with patch.dict(os.environ, {"MCP_CLI_CONFIGS": "dynamic_tools"}, clear=True):
            self._reset_config()
            config = get_config()
            assert config.mcp_server_register_dynamic_tools_on_startup is True
            assert config.mcp_server_register_dynamic_prompts_on_startup is False
            assert config.tool_config.enable_predictive_tools is True
            assert config.tool_config.enable_gdrive_tools is False
            assert config.tool_config.enable_jira_tools is False
            assert config.tool_config.enable_confluence_tools is False
            assert config.tool_config.enable_microsoft_graph_tools is False
            assert config.tool_config.enable_perplexity_tools is False
            assert config.tool_config.enable_tavily_tools is False
            self._reset_config()

    def test_mcp_cli_configs_single_option_enables_tool_field(self) -> None:
        """Option in MCP_CLI_CONFIGS with no individual env overrides tool field to True."""
        with patch.dict(os.environ, {"MCP_CLI_CONFIGS": "gdrive"}, clear=True):
            self._reset_config()
            config = get_config()
            assert config.tool_config.enable_gdrive_tools is True
            assert config.tool_config.enable_jira_tools is False
            assert config.tool_config.enable_confluence_tools is False
            assert config.tool_config.enable_microsoft_graph_tools is False
            assert config.tool_config.enable_perplexity_tools is False
            assert config.tool_config.enable_tavily_tools is False
            self._reset_config()

    def test_mcp_cli_configs_multiple_options(self) -> None:
        """Multiple options in MCP_CLI_CONFIGS each override their field to True when at default."""
        with patch.dict(
            os.environ,
            {
                "MCP_CLI_CONFIGS": (
                    "dynamic_tools,dynamic_prompts,gdrive,jira,"
                    "confluence,microsoft_graph,predictive,perplexity,tavily"
                )
            },
            clear=True,
        ):
            self._reset_config()
            config = get_config()
            assert config.mcp_server_register_dynamic_tools_on_startup is True
            assert config.mcp_server_register_dynamic_prompts_on_startup is True
            assert config.tool_config.enable_predictive_tools is True
            assert config.tool_config.enable_gdrive_tools is True
            assert config.tool_config.enable_jira_tools is True
            assert config.tool_config.enable_confluence_tools is True
            assert config.tool_config.enable_microsoft_graph_tools is True
            assert config.tool_config.enable_perplexity_tools is True
            assert config.tool_config.enable_tavily_tools is True
            self._reset_config()

    def test_mcp_cli_configs_case_insensitive(self) -> None:
        """MCP_CLI_CONFIGS option names are case-insensitive."""
        with patch.dict(os.environ, {"MCP_CLI_CONFIGS": "GDRIVE,Jira"}, clear=True):
            self._reset_config()
            config = get_config()
            assert config.tool_config.enable_gdrive_tools is True
            assert config.tool_config.enable_jira_tools is True
            assert config.tool_config.enable_confluence_tools is False
            assert config.tool_config.enable_microsoft_graph_tools is False
            assert config.tool_config.enable_perplexity_tools is False
            assert config.tool_config.enable_tavily_tools is False
            self._reset_config()

    def test_mcp_cli_configs_whitespace_stripped(self) -> None:
        """Whitespace around options in MCP_CLI_CONFIGS is stripped."""
        with patch.dict(os.environ, {"MCP_CLI_CONFIGS": "  gdrive  ,  jira  "}, clear=True):
            self._reset_config()
            config = get_config()
            assert config.tool_config.enable_gdrive_tools is True
            assert config.tool_config.enable_jira_tools is True
            assert config.tool_config.enable_confluence_tools is False
            assert config.tool_config.enable_microsoft_graph_tools is False
            assert config.tool_config.enable_perplexity_tools is False
            assert config.tool_config.enable_tavily_tools is False
            self._reset_config()

    def test_individual_env_takes_precedence_unprefixed(self) -> None:
        """When individual env var is set, MCP_CLI_CONFIGS does not override that field."""
        with patch.dict(
            os.environ,
            {"MCP_CLI_CONFIGS": "gdrive,jira", "ENABLE_GDRIVE_TOOLS": "false"},
            clear=True,
        ):
            self._reset_config()
            config = get_config()
            assert config.tool_config.enable_gdrive_tools is False
            assert config.tool_config.enable_jira_tools is True
            assert config.tool_config.enable_confluence_tools is False
            assert config.tool_config.enable_microsoft_graph_tools is False
            assert config.tool_config.enable_perplexity_tools is False
            assert config.tool_config.enable_tavily_tools is False
            self._reset_config()

    def test_individual_env_takes_precedence_runtime_param_prefix(self) -> None:
        """When MLOPS_RUNTIME_PARAM_* env is set, MCP_CLI_CONFIGS does not override that field."""
        with patch.dict(
            os.environ,
            {"MCP_CLI_CONFIGS": "gdrive", "MLOPS_RUNTIME_PARAM_ENABLE_GDRIVE_TOOLS": "false"},
            clear=True,
        ):
            self._reset_config()
            config = get_config()
            assert config.tool_config.enable_gdrive_tools is False
            assert config.tool_config.enable_jira_tools is False
            assert config.tool_config.enable_confluence_tools is False
            assert config.tool_config.enable_microsoft_graph_tools is False
            assert config.tool_config.enable_perplexity_tools is False
            assert config.tool_config.enable_tavily_tools is False
            self._reset_config()

    def test_individual_env_true_with_mcp_cli_configs(self) -> None:
        """Env true for field not in list unchanged; when in list, env wins."""
        with patch.dict(
            os.environ,
            {"MCP_CLI_CONFIGS": "jira", "ENABLE_GDRIVE_TOOLS": "true"},
            clear=True,
        ):
            self._reset_config()
            config = get_config()
            assert config.tool_config.enable_gdrive_tools is True
            assert config.tool_config.enable_jira_tools is True
            assert config.tool_config.enable_confluence_tools is False
            assert config.tool_config.enable_microsoft_graph_tools is False
            assert config.tool_config.enable_perplexity_tools is False
            assert config.tool_config.enable_tavily_tools is False
            self._reset_config()

    def test_option_not_in_mcp_cli_configs_never_overridden(self) -> None:
        """Options not in MCP_CLI_CONFIGS stay at default; only listed ones set True."""
        with patch.dict(os.environ, {"MCP_CLI_CONFIGS": "gdrive"}, clear=True):
            self._reset_config()
            config = get_config()
            assert config.tool_config.enable_gdrive_tools is True
            assert config.tool_config.enable_jira_tools is False
            assert config.tool_config.enable_confluence_tools is False
            assert config.mcp_server_register_dynamic_prompts_on_startup is False
            assert config.tool_config.enable_perplexity_tools is False
            assert config.tool_config.enable_tavily_tools is False
            self._reset_config()

    def test_predictive_default_true_stays_true_when_in_mcp_cli_configs(self) -> None:
        """Predictive has default True; when in MCP_CLI_CONFIGS and no env, stays True."""
        with patch.dict(os.environ, {"MCP_CLI_CONFIGS": "predictive"}, clear=True):
            self._reset_config()
            config = get_config()
            assert config.tool_config.enable_predictive_tools is True
            assert config.tool_config.enable_gdrive_tools is False
            assert config.tool_config.enable_jira_tools is False
            assert config.tool_config.enable_confluence_tools is False
            assert config.tool_config.enable_microsoft_graph_tools is False
            assert config.tool_config.enable_perplexity_tools is False
            assert config.tool_config.enable_tavily_tools is False
            self._reset_config()

    def test_dynamic_prompts_override_when_in_mcp_cli_configs(self) -> None:
        """dynamic_prompts default False; when in MCP_CLI_CONFIGS and no env, becomes True."""
        with patch.dict(os.environ, {"MCP_CLI_CONFIGS": "dynamic_prompts"}, clear=True):
            self._reset_config()
            config = get_config()
            assert config.mcp_server_register_dynamic_prompts_on_startup is True
            assert config.tool_config.enable_predictive_tools is True
            assert config.tool_config.enable_gdrive_tools is False
            assert config.tool_config.enable_jira_tools is False
            assert config.tool_config.enable_confluence_tools is False
            assert config.tool_config.enable_microsoft_graph_tools is False
            assert config.tool_config.enable_perplexity_tools is False
            assert config.tool_config.enable_tavily_tools is False
            self._reset_config()

    def test_individual_env_false_for_option_in_mcp_cli_configs_respected(self) -> None:
        """ENABLE_JIRA_TOOLS=false with jira in MCP_CLI_CONFIGS: env wins (no override)."""
        with patch.dict(
            os.environ,
            {"MCP_CLI_CONFIGS": "jira,gdrive", "ENABLE_JIRA_TOOLS": "false"},
            clear=True,
        ):
            self._reset_config()
            config = get_config()
            assert config.tool_config.enable_jira_tools is False
            assert config.tool_config.enable_gdrive_tools is True
            assert config.tool_config.enable_confluence_tools is False
            assert config.tool_config.enable_microsoft_graph_tools is False
            assert config.tool_config.enable_perplexity_tools is False
            assert config.tool_config.enable_tavily_tools is False
            self._reset_config()

    def test_mcp_cli_configs_value_comes_from_config_loading(self) -> None:
        """mcp_cli_configs is populated from env (MCP_CLI_CONFIGS) like other config fields."""
        with patch.dict(os.environ, {"MCP_CLI_CONFIGS": "gdrive,jira"}, clear=True):
            self._reset_config()
            config = get_config()
            assert config.mcp_cli_configs.strip().lower() == "gdrive,jira"
            assert config.tool_config.enable_gdrive_tools is True
            assert config.tool_config.enable_jira_tools is True
            assert config.tool_config.enable_confluence_tools is False
            assert config.tool_config.enable_microsoft_graph_tools is False
            assert config.tool_config.enable_perplexity_tools is False
            assert config.tool_config.enable_tavily_tools is False
            self._reset_config()
