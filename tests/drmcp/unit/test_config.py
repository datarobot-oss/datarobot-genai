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
