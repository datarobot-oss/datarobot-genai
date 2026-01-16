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

"""Tests for tool configuration and enablement logic."""

import os
from unittest.mock import MagicMock
from unittest.mock import patch

from datarobot_genai.drmcp.core.config import MCPServerConfig
from datarobot_genai.drmcp.core.tool_config import TOOL_CONFIGS
from datarobot_genai.drmcp.core.tool_config import ToolType
from datarobot_genai.drmcp.core.tool_config import get_tool_enable_config_name
from datarobot_genai.drmcp.core.tool_config import is_tool_enabled


class TestToolType:
    """Test ToolType enum."""

    def test_tool_type_values(self) -> None:
        """Test that ToolType enum has expected values."""
        assert ToolType.PREDICTIVE.value == "predictive"
        assert ToolType.JIRA.value == "jira"
        assert ToolType.CONFLUENCE.value == "confluence"
        assert ToolType.MICROSOFT_GRAPH.value == "microsoft_graph"

    def test_tool_type_is_string_enum(self) -> None:
        """Test that ToolType is a string enum."""
        # ToolType inherits from str and Enum, so the value is a string
        assert ToolType.PREDICTIVE.value == "predictive"
        # The enum member itself can be compared to string
        assert ToolType.PREDICTIVE == "predictive"


class TestToolConfigs:
    """Test TOOL_CONFIGS registry."""

    def test_all_tool_types_have_configs(self) -> None:
        """Test that all ToolType values have corresponding configs."""
        for tool_type in ToolType:
            assert tool_type in TOOL_CONFIGS

    def test_predictive_tool_config(self) -> None:
        """Test predictive tool configuration."""
        config = TOOL_CONFIGS[ToolType.PREDICTIVE]
        assert config["name"] == "predictive"
        assert config["oauth_check"] is None
        assert config["directory"] == "predictive"
        assert config["package_prefix"] == "datarobot_genai.drmcp.tools.predictive"
        assert config["config_field_name"] == "enable_predictive_tools"

    def test_jira_tool_config(self) -> None:
        """Test Jira tool configuration."""
        config = TOOL_CONFIGS[ToolType.JIRA]
        assert config["name"] == "jira"
        assert config["oauth_check"] is not None
        assert config["directory"] == "jira"
        assert config["package_prefix"] == "datarobot_genai.drmcp.tools.jira"
        assert config["config_field_name"] == "enable_jira_tools"

    def test_confluence_tool_config(self) -> None:
        """Test Confluence tool configuration."""
        config = TOOL_CONFIGS[ToolType.CONFLUENCE]
        assert config["name"] == "confluence"
        assert config["oauth_check"] is not None
        assert config["directory"] == "confluence"
        assert config["package_prefix"] == "datarobot_genai.drmcp.tools.confluence"
        assert config["config_field_name"] == "enable_confluence_tools"

    def test_microsoft_graph_tool_config(self) -> None:
        """Test Microsoft Graph tool configuration."""
        config = TOOL_CONFIGS[ToolType.MICROSOFT_GRAPH]
        assert config["name"] == "microsoft_graph"
        assert config["oauth_check"] is not None
        assert config["directory"] == "microsoft_graph"
        assert config["package_prefix"] == "datarobot_genai.drmcp.tools.microsoft_graph"
        assert config["config_field_name"] == "enable_microsoft_graph_tools"

    def test_jira_oauth_check_callable(self) -> None:
        """Test that Jira OAuth check is a callable."""
        config = TOOL_CONFIGS[ToolType.JIRA]
        assert callable(config["oauth_check"])

        # Test that it calls the config method
        mock_config = MagicMock(spec=MCPServerConfig)
        mock_tool_config = MagicMock()
        mock_tool_config.is_atlassian_oauth_configured = True
        mock_config.tool_config = mock_tool_config
        assert config["oauth_check"](mock_config) is True

        mock_tool_config.is_atlassian_oauth_configured = False
        assert config["oauth_check"](mock_config) is False

    def test_confluence_oauth_check_callable(self) -> None:
        """Test that Confluence OAuth check is a callable."""
        config = TOOL_CONFIGS[ToolType.CONFLUENCE]
        assert callable(config["oauth_check"])

        # Test that it calls the config method
        mock_config = MagicMock(spec=MCPServerConfig)
        mock_tool_config = MagicMock()
        mock_tool_config.is_atlassian_oauth_configured = True
        mock_config.tool_config = mock_tool_config
        assert config["oauth_check"](mock_config) is True

        mock_tool_config.is_atlassian_oauth_configured = False
        assert config["oauth_check"](mock_config) is False

    def test_microsoft_graph_oauth_check_callable(self) -> None:
        """Test that Microsoft Graph OAuth check is a callable."""
        config = TOOL_CONFIGS[ToolType.MICROSOFT_GRAPH]
        assert callable(config["oauth_check"])

        # Test that it calls the config method
        mock_config = MagicMock(spec=MCPServerConfig)
        mock_tool_config = MagicMock()
        mock_tool_config.is_microsoft_oauth_configured = True
        mock_config.tool_config = mock_tool_config
        assert config["oauth_check"](mock_config) is True

        mock_tool_config.is_microsoft_oauth_configured = False
        assert config["oauth_check"](mock_config) is False


class TestGetToolEnableConfigName:
    """Test get_tool_enable_config_name function."""

    def test_predictive_config_name(self) -> None:
        """Test config name for predictive tools."""
        assert get_tool_enable_config_name(ToolType.PREDICTIVE) == "enable_predictive_tools"

    def test_jira_config_name(self) -> None:
        """Test config name for Jira tools."""
        assert get_tool_enable_config_name(ToolType.JIRA) == "enable_jira_tools"

    def test_confluence_config_name(self) -> None:
        """Test config name for Confluence tools."""
        assert get_tool_enable_config_name(ToolType.CONFLUENCE) == "enable_confluence_tools"

    def test_microsoft_graph_config_name(self) -> None:
        """Test config name for Microsoft Graph tools."""
        assert (
            get_tool_enable_config_name(ToolType.MICROSOFT_GRAPH) == "enable_microsoft_graph_tools"
        )


class TestIsToolEnabled:
    """Test is_tool_enabled function."""

    def test_predictive_tool_enabled(self) -> None:
        """Test predictive tool when enabled."""
        with patch.dict(os.environ, {"ENABLE_PREDICTIVE_TOOLS": "true"}, clear=False):
            config = MCPServerConfig()
            assert is_tool_enabled(ToolType.PREDICTIVE, config) is True

    def test_predictive_tool_disabled(self) -> None:
        """Test predictive tool when disabled."""
        with patch.dict(os.environ, {"ENABLE_PREDICTIVE_TOOLS": "false"}, clear=False):
            config = MCPServerConfig()
            assert is_tool_enabled(ToolType.PREDICTIVE, config) is False

    def test_predictive_tool_default_enabled(self) -> None:
        """Test predictive tool default is enabled."""
        with patch.dict(os.environ, clear=True):
            config = MCPServerConfig(_env_file=None)
            assert is_tool_enabled(ToolType.PREDICTIVE, config) is True

    def test_jira_tool_enabled_with_oauth(self) -> None:
        """Test Jira tool when enabled and OAuth is configured."""
        with patch.dict(
            os.environ,
            {
                "ENABLE_JIRA_TOOLS": "true",
                "IS_ATLASSIAN_OAUTH_PROVIDER_CONFIGURED": "true",
            },
            clear=False,
        ):
            config = MCPServerConfig()
            assert is_tool_enabled(ToolType.JIRA, config) is True

    def test_jira_tool_enabled_without_oauth(self) -> None:
        """Test Jira tool when enabled but OAuth is not configured."""
        with patch.dict(os.environ, {"ENABLE_JIRA_TOOLS": "true"}, clear=False):
            config = MCPServerConfig()
            # No OAuth configured
            assert is_tool_enabled(ToolType.JIRA, config) is False

    def test_jira_tool_enabled_with_env_vars(self) -> None:
        """Test Jira tool when enabled and OAuth via env vars."""
        with patch.dict(
            os.environ,
            {
                "ENABLE_JIRA_TOOLS": "true",
                "ATLASSIAN_CLIENT_ID": "test_id",
                "ATLASSIAN_CLIENT_SECRET": "test_secret",
            },
            clear=False,
        ):
            config = MCPServerConfig()
            assert is_tool_enabled(ToolType.JIRA, config) is True

    def test_jira_tool_disabled(self) -> None:
        """Test Jira tool when disabled."""
        with patch.dict(
            os.environ,
            {
                "ENABLE_JIRA_TOOLS": "false",
                "IS_ATLASSIAN_OAUTH_PROVIDER_CONFIGURED": "true",
            },
            clear=False,
        ):
            config = MCPServerConfig()
            # Even with OAuth configured, tool should be disabled
            assert is_tool_enabled(ToolType.JIRA, config) is False

    def test_confluence_tool_enabled_with_oauth(self) -> None:
        """Test Confluence tool when enabled and OAuth is configured."""
        with patch.dict(
            os.environ,
            {
                "ENABLE_CONFLUENCE_TOOLS": "true",
                "IS_ATLASSIAN_OAUTH_PROVIDER_CONFIGURED": "true",
            },
            clear=False,
        ):
            config = MCPServerConfig()
            assert is_tool_enabled(ToolType.CONFLUENCE, config) is True

    def test_confluence_tool_enabled_without_oauth(self) -> None:
        """Test Confluence tool when enabled but OAuth is not configured."""
        with patch.dict(os.environ, {"ENABLE_CONFLUENCE_TOOLS": "true"}, clear=False):
            config = MCPServerConfig()
            # No OAuth configured
            assert is_tool_enabled(ToolType.CONFLUENCE, config) is False

    def test_confluence_tool_enabled_with_env_vars(self) -> None:
        """Test Confluence tool when enabled and OAuth via env vars."""
        with patch.dict(
            os.environ,
            {
                "ENABLE_CONFLUENCE_TOOLS": "true",
                "ATLASSIAN_CLIENT_ID": "test_id",
                "ATLASSIAN_CLIENT_SECRET": "test_secret",
            },
            clear=False,
        ):
            config = MCPServerConfig()
            assert is_tool_enabled(ToolType.CONFLUENCE, config) is True

    def test_confluence_tool_disabled(self) -> None:
        """Test Confluence tool when disabled."""
        with patch.dict(
            os.environ,
            {
                "ENABLE_CONFLUENCE_TOOLS": "false",
                "IS_ATLASSIAN_OAUTH_PROVIDER_CONFIGURED": "true",
            },
            clear=False,
        ):
            config = MCPServerConfig()
            # Even with OAuth configured, tool should be disabled
            assert is_tool_enabled(ToolType.CONFLUENCE, config) is False

    def test_microsoft_graph_tool_enabled_with_oauth(self) -> None:
        """Test Microsoft Graph tool when enabled and OAuth is configured."""
        with patch.dict(
            os.environ,
            {
                "ENABLE_MICROSOFT_GRAPH_TOOLS": "true",
                "IS_MICROSOFT_OAUTH_PROVIDER_CONFIGURED": "true",
            },
            clear=False,
        ):
            config = MCPServerConfig()
            assert is_tool_enabled(ToolType.MICROSOFT_GRAPH, config) is True

    def test_microsoft_graph_tool_enabled_without_oauth(self) -> None:
        """Test Microsoft Graph tool when enabled but OAuth is not configured."""
        with patch.dict(os.environ, {"ENABLE_MICROSOFT_GRAPH_TOOLS": "true"}, clear=False):
            config = MCPServerConfig()
            # No OAuth configured
            assert is_tool_enabled(ToolType.MICROSOFT_GRAPH, config) is False

    def test_microsoft_graph_tool_enabled_with_env_vars(self) -> None:
        """Test Microsoft Graph tool when enabled and OAuth via env vars."""
        with patch.dict(
            os.environ,
            {
                "ENABLE_MICROSOFT_GRAPH_TOOLS": "true",
                "MICROSOFT_CLIENT_ID": "test_id",
                "MICROSOFT_CLIENT_SECRET": "test_secret",
            },
            clear=False,
        ):
            config = MCPServerConfig()
            assert is_tool_enabled(ToolType.MICROSOFT_GRAPH, config) is True

    def test_microsoft_graph_tool_disabled(self) -> None:
        """Test Microsoft Graph tool when disabled."""
        with patch.dict(
            os.environ,
            {
                "ENABLE_MICROSOFT_GRAPH_TOOLS": "false",
                "IS_MICROSOFT_OAUTH_PROVIDER_CONFIGURED": "true",
            },
            clear=False,
        ):
            config = MCPServerConfig()
            # Even with OAuth configured, tool should be disabled
            assert is_tool_enabled(ToolType.MICROSOFT_GRAPH, config) is False
