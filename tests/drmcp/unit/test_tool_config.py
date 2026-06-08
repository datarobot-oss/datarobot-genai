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
from unittest.mock import patch

from datarobot_genai.drmcp.core.config import MCPServerConfig
from datarobot_genai.drmcp.core.config import MCPToolConfig
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
        assert ToolType.USE_CASE.value == "use_case"
        assert ToolType.VDB.value == "vdb"
        assert ToolType.CODE_EXECUTION.value == "code_execution"
        assert ToolType.OPTIMIZATION.value == "optimization"
        assert ToolType.WORKLOAD.value == "workload"

    def test_tool_type_is_string_enum(self) -> None:
        """Test that ToolType is a string enum."""
        assert ToolType.PREDICTIVE.value == "predictive"
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
        assert config["directory"] == "predictive"
        assert config["package_prefix"] == "datarobot_genai.drtools.predictive"
        assert config["config_field_name"] == "enable_predictive_tools"

    def test_jira_tool_config(self) -> None:
        """Test Jira tool configuration."""
        config = TOOL_CONFIGS[ToolType.JIRA]
        assert config["name"] == "jira"
        assert config["directory"] == "jira"
        assert config["package_prefix"] == "datarobot_genai.drtools.jira"
        assert config["config_field_name"] == "enable_jira_tools"

    def test_confluence_tool_config(self) -> None:
        """Test Confluence tool configuration."""
        config = TOOL_CONFIGS[ToolType.CONFLUENCE]
        assert config["name"] == "confluence"
        assert config["directory"] == "confluence"
        assert config["package_prefix"] == "datarobot_genai.drtools.confluence"
        assert config["config_field_name"] == "enable_confluence_tools"

    def test_microsoft_graph_tool_config(self) -> None:
        """Test Microsoft Graph tool configuration."""
        config = TOOL_CONFIGS[ToolType.MICROSOFT_GRAPH]
        assert config["name"] == "microsoft_graph"
        assert config["directory"] == "microsoft_graph"
        assert config["package_prefix"] == "datarobot_genai.drtools.microsoft_graph"
        assert config["config_field_name"] == "enable_microsoft_graph_tools"

    def test_workload_tool_config(self) -> None:
        """Test workload tool configuration."""
        config = TOOL_CONFIGS[ToolType.WORKLOAD]
        assert config["name"] == "workload"
        assert config["directory"] == "workload"
        assert config["package_prefix"] == "datarobot_genai.drtools.workload"
        assert config["config_field_name"] == "enable_workload_tools"


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

    def test_workload_config_name(self) -> None:
        """Test config name for workload tools."""
        assert get_tool_enable_config_name(ToolType.WORKLOAD) == "enable_workload_tools"


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

    def test_predictive_tool_default_disabled(self) -> None:
        """Test predictive tool default is disabled."""
        with patch.dict(os.environ, clear=True):
            config = MCPServerConfig(_env_file=None, tool_config=MCPToolConfig(_env_file=None))
            assert is_tool_enabled(ToolType.PREDICTIVE, config) is False

    def test_jira_tool_enabled(self) -> None:
        """Test Jira tool when enabled."""
        with patch.dict(os.environ, {"ENABLE_JIRA_TOOLS": "true"}, clear=False):
            config = MCPServerConfig()
            assert is_tool_enabled(ToolType.JIRA, config) is True

    def test_jira_tool_disabled(self) -> None:
        """Test Jira tool when disabled."""
        with patch.dict(os.environ, {"ENABLE_JIRA_TOOLS": "false"}, clear=False):
            config = MCPServerConfig()
            assert is_tool_enabled(ToolType.JIRA, config) is False

    def test_confluence_tool_enabled(self) -> None:
        """Test Confluence tool when enabled."""
        with patch.dict(os.environ, {"ENABLE_CONFLUENCE_TOOLS": "true"}, clear=False):
            config = MCPServerConfig()
            assert is_tool_enabled(ToolType.CONFLUENCE, config) is True

    def test_microsoft_graph_tool_enabled(self) -> None:
        """Test Microsoft Graph tool when enabled."""
        with patch.dict(os.environ, {"ENABLE_MICROSOFT_GRAPH_TOOLS": "true"}, clear=False):
            config = MCPServerConfig()
            assert is_tool_enabled(ToolType.MICROSOFT_GRAPH, config) is True

    def test_workload_tool_enabled(self) -> None:
        """Test workload tool when enabled."""
        with patch.dict(os.environ, {"ENABLE_WORKLOAD_TOOLS": "true"}, clear=False):
            config = MCPServerConfig()
            assert is_tool_enabled(ToolType.WORKLOAD, config) is True

    def test_workload_tool_disabled(self) -> None:
        """Test workload tool when disabled."""
        with patch.dict(os.environ, {"ENABLE_WORKLOAD_TOOLS": "false"}, clear=False):
            config = MCPServerConfig()
            assert is_tool_enabled(ToolType.WORKLOAD, config) is False

    def test_workload_tool_default_disabled(self) -> None:
        """Test workload tool default is disabled."""
        with patch.dict(os.environ, clear=True):
            config = MCPServerConfig(_env_file=None, tool_config=MCPToolConfig(_env_file=None))
            assert is_tool_enabled(ToolType.WORKLOAD, config) is False
