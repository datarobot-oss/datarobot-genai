# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from unittest.mock import patch

import pytest
from datarobot.models.genai.agent.auth import set_authorization_context

from datarobot_genai.core.mcp.common import MCPConfig


class TestMCPConfig:
    """Test MCP configuration management."""

    @pytest.fixture(autouse=True)
    def empty_agent_auth_context(self):
        set_authorization_context({})

    def test_mcp_config_without_configuration(self):
        """Test MCP config when no environment variables are set."""
        with patch.dict(os.environ, {}, clear=True):
            config = MCPConfig()
            assert config.external_mcp_url is None
            assert config.mcp_deployment_id is None
            assert config.server_config is None

    def test_mcp_config_with_external_url(self):
        """Test MCP config with external URL."""
        test_url = "https://mcp-server.example.com/mcp"
        with patch.dict(os.environ, {"EXTERNAL_MCP_URL": test_url}, clear=True):
            config = MCPConfig()
            assert config.external_mcp_url == test_url
            assert config.server_config is not None
            assert config.server_config["url"] == test_url
            assert config.server_config["headers"] == {}

    def test_mcp_config_with_datarobot_deployment_id(self, agent_auth_context_data):
        """Test MCP config with DataRobot deployment ID."""
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "test-api-key"
        secret_key = "my-secret-key"

        # When the agent is initialized, it sets the authorization context for the
        # process, so subsequent tools and MCP calls receive it via a dedicated header.
        set_authorization_context(agent_auth_context_data)

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": api_base,
                "DATAROBOT_API_TOKEN": api_key,
                "SESSION_SECRET_KEY": secret_key,
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.mcp_deployment_id == deployment_id
            assert config.server_config is not None
            assert (
                config.server_config["url"]
                == f"{api_base}/deployments/{deployment_id}/directAccess/mcp"
            )
            assert config.server_config["headers"]["Authorization"] == f"Bearer {api_key}"

            # Verify the authorization context header is propagated correctly
            # from the Agent to the MCP Server and the header can be decoded.
            jwt_token = config.server_config["headers"]["X-DataRobot-Authorization-Context"]
            decoded_auth_context = config.auth_context_handler.decode(jwt_token)
            assert agent_auth_context_data == decoded_auth_context

    def test_mcp_config_with_datarobot_deployment_id_and_bearer_token(self):
        """Test MCP config with DataRobot deployment ID and Bearer token already formatted."""
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "Bearer test-api-key"

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": api_base,
                "DATAROBOT_API_TOKEN": api_key,
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.server_config["headers"]["Authorization"] == api_key

        # When authorization context is empty for the Agent, the header should not
        # be propagated to the MCP Server.
        assert "X-DataRobot-Authorization-Context" not in config.server_config["headers"]

    def test_mcp_config_with_datarobot_deployment_id_no_api_key(self):
        """Test MCP config with DataRobot deployment ID but no API key."""
        deployment_id = "abc123def456789012345678"

        with patch.dict(os.environ, {"MCP_DEPLOYMENT_ID": deployment_id}, clear=True):
            config = MCPConfig()
            assert config.server_config is None

    def test_mcp_config_with_datarobot_deployment_id_no_deployment_id(self):
        """Test MCP config with API key but no deployment ID."""
        api_key = "test-api-key"

        with patch.dict(os.environ, {"DATAROBOT_API_TOKEN": api_key}, clear=True):
            config = MCPConfig()
            assert config.server_config is None

    def test_mcp_config_url_construction_with_trailing_slash(self):
        """Test URL construction when api_base has trailing slash."""
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2/"
        api_key = "test-api-key"

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": api_base,
                "DATAROBOT_API_TOKEN": api_key,
            },
            clear=True,
        ):
            config = MCPConfig()
            expected_url = "https://app.datarobot.com/api/v2/deployments/abc123def456789012345678/directAccess/mcp"
            assert config.server_config["url"] == expected_url

    def test_mcp_config_priority_external_over_deployment(self):
        """Test that EXTERNAL_MCP_URL takes priority over MCP_DEPLOYMENT_ID."""
        external_url = "https://external-mcp.com/mcp"
        deployment_id = "abc123def456789012345678"
        api_key = "test-api-key"

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": external_url,
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_API_TOKEN": api_key,
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.server_config["url"] == external_url
            assert config.server_config["headers"] == {}

    def test_mcp_config_with_external_headers(self):
        test_url = "https://mcp-server.example.com/mcp"
        headers = {"X-Test": "value", "Authorization": "Bearer fake_api_key"}
        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
                "EXTERNAL_MCP_HEADERS": json.dumps(headers),
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.server_config["url"] == test_url
            assert config.server_config["headers"] == headers

    def test_mcp_config_with_external_headers_invalid_json(self):
        test_url = "https://mcp-server.example.com/mcp"
        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
                "EXTERNAL_MCP_HEADERS": "not-a-json",
            },
            clear=True,
        ):
            with pytest.raises(json.JSONDecodeError):
                MCPConfig()

    def test_mcp_config_with_external_transport(self):
        test_url = "https://mcp-server.example.com/mcp"
        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
                "EXTERNAL_MCP_TRANSPORT": "custom-transport",
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.external_mcp_transport == "custom-transport"
            assert config.server_config["url"] == test_url

    def test_mcp_config_with_direct_params(self):
        deployment_id = "abc123def456789012345678"
        api_base = "https://custom.api/v2"
        api_key = "fake_api_key"
        with patch.dict(
            os.environ,
            {"MCP_DEPLOYMENT_ID": deployment_id},
            clear=True,
        ):
            config = MCPConfig(api_base=api_base, api_key=api_key)
            expected_url = f"{api_base}/deployments/{deployment_id}/directAccess/mcp"
            assert config.server_config["url"] == expected_url
            assert config.server_config["headers"]["Authorization"] == f"Bearer {api_key}"

    def test_mcp_config_with_bearer_only_api_key(self):
        deployment_id = "abc123def456789012345678"
        api_base = "https://custom.api/v2"
        api_key = "Bearer fake_api_key"
        with patch.dict(
            os.environ,
            {"MCP_DEPLOYMENT_ID": deployment_id},
            clear=True,
        ):
            config = MCPConfig(api_base=api_base, api_key=api_key)
            assert config.server_config["headers"]["Authorization"] == "Bearer fake_api_key"

    def test_mcp_config_with_whitespace_api_key(self):
        deployment_id = "abc123def456789012345678"
        api_base = "https://custom.api/v2"
        api_key = "fake_api_key"
        with patch.dict(
            os.environ,
            {"MCP_DEPLOYMENT_ID": deployment_id},
            clear=True,
        ):
            config = MCPConfig(api_base=api_base, api_key=api_key)
            assert config.server_config["headers"]["Authorization"] == f"Bearer {api_key}"

    def test_mcp_config_none_when_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            config = MCPConfig()
            assert config.server_config is None

    def test_mcp_config_none_when_all_empty(self):
        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": "",
                "MCP_DEPLOYMENT_ID": "",
                "DATAROBOT_API_TOKEN": "",
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.server_config is None
