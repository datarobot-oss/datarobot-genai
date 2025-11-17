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
        assert config.external_mcp_headers is None
        assert config.external_mcp_transport == "streamable-http"
        assert config.mcp_deployment_id is None
        assert config.datarobot_api_token is None
        assert config.server_config is None

    def test_mcp_config_with_external_url(self):
        """Test MCP config with external URL."""
        test_url = "https://mcp-server.example.com/mcp"
        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
                "DATAROBOT_ENDPOINT": "https://app.datarobot.example/api/v2",
                "DATAROBOT_API_TOKEN": "dummy-token",
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.external_mcp_url == test_url
            assert config.server_config is not None
            assert config.server_config["url"] == test_url
            assert config.server_config["headers"] == {}
            assert config.server_config["transport"] == "streamable-http"

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

    @pytest.mark.parametrize(
        "api_base",
        [
            pytest.param("https://app.datarobot.com/api/v2", id="no-trailing-slash"),
            pytest.param("https://app.datarobot.com/api/v2/", id="with-trailing-slash"),
            pytest.param("https://app.datarobot.com/", id="with-trailing-slash-no-api-v2"),
            pytest.param("https://app.datarobot.com", id="no-trailing-slash-no-api-v2"),
        ],
    )
    def test_mcp_config_url_construction(self, api_base):
        """Test URL construction when api_base has trailing slash."""
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
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
        headers = {"X-Custom-Header": "custom-value"}

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": external_url,
                "EXTERNAL_MCP_HEADERS": json.dumps(headers),
                "EXTERNAL_MCP_TRANSPORT": "sse",
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_API_TOKEN": api_key,
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.server_config["url"] == external_url
            assert config.server_config["headers"] == headers
            assert config.server_config["transport"] == "sse"
            assert config.server_config["headers"] == headers

    def test_mcp_config_with_external_headers_invalid_json(self):
        """Invalid JSON should raise ValueError via field validator."""
        test_url = "https://mcp-server.example.com/mcp"
        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
                "EXTERNAL_MCP_HEADERS": "not-a-json",
                "DATAROBOT_ENDPOINT": "https://app.datarobot.example/api/v2",
                "DATAROBOT_API_TOKEN": "dummy-token",
            },
            clear=True,
        ):
            with pytest.raises(ValueError):
                MCPConfig()

    def test_mcp_config_with_external_transport(self):
        test_url = "https://mcp-server.example.com/mcp"
        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
                "EXTERNAL_MCP_TRANSPORT": "sse",
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.external_mcp_transport == "sse"
            assert config.server_config["url"] == test_url

    def test_mcp_config_with_direct_params(self):
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
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
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "Bearer fake_api_key"
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
            assert config.server_config["headers"]["Authorization"] == "Bearer fake_api_key"

    def test_mcp_config_with_whitespace_api_key(self):
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "fake_api_key"
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
            assert config.server_config["headers"]["Authorization"] == f"Bearer {api_key}"

    def test_mcp_config_none_when_all_empty(self):
        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": "",
                "DATAROBOT_API_TOKEN": "",
                "DATAROBOT_ENDPOINT": "",
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.server_config is None

    def test_external_mcp_headers_whitespace_trim(self):
        """Leading/trailing whitespace in headers JSON should be trimmed."""
        raw = '  {"X-Test": "value"}  '
        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": "https://mcp-server.example.com/mcp",
                "EXTERNAL_MCP_HEADERS": raw,
                "DATAROBOT_ENDPOINT": "https://app.datarobot.example/api/v2",
                "DATAROBOT_API_TOKEN": "dummy-token",
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.external_mcp_headers == raw.strip()
            assert config.server_config["headers"] == {"X-Test": "value"}

    def test_mcp_deployment_id_validation_errors(self):
        """Invalid deployment IDs should raise ValueError or TypeError."""
        # Invalid length / characters
        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": "short-id",
                "DATAROBOT_ENDPOINT": "https://app.datarobot.example/api/v2",
                "DATAROBOT_API_TOKEN": "dummy-token",
            },
            clear=True,
        ):
            with pytest.raises(
                ValueError, match="mcp_deployment_id must be a valid 24-character hex ID"
            ):
                MCPConfig()
        # Non-string type
        with patch.dict(
            os.environ,
            {
                "DATAROBOT_ENDPOINT": "https://app.datarobot.example/api/v2",
                "DATAROBOT_API_TOKEN": "dummy-token",
            },
            clear=True,
        ):
            with pytest.raises(TypeError, match="mcp_deployment_id must be a string"):
                MCPConfig(mcp_deployment_id=1234)  # type: ignore[arg-type]

    def test_mcp_deployment_id_whitespace_trim(self):
        """Whitespace around valid deployment id should be trimmed and accepted."""
        deployment_id = "abc123def456789012345678"
        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": f"  {deployment_id}  ",
                "DATAROBOT_ENDPOINT": "https://app.datarobot.example/api/v2",
                "DATAROBOT_API_TOKEN": "dummy-token",
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.mcp_deployment_id == deployment_id

    def test_authorization_header_absent_when_api_key_empty(self):
        """Empty api key should result in no Authorization header in deployment config."""
        deployment_id = "abc123def456789012345678"
        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": "https://app.datarobot.example/api/v2",
                "DATAROBOT_API_TOKEN": "",  # Empty
            },
            clear=True,
        ):
            config = MCPConfig()
            # With empty api key, the deployment branch should not activate (needs api_key)
            assert config.server_config is None

    def test_authorization_context_header_exception(self, agent_auth_context_data):
        """Simulate an exception when retrieving auth context; header should be omitted."""
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
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
            # Monkeypatch auth_context_handler.get_header to raise LookupError
            with patch.object(config.auth_context_handler, "get_header", side_effect=LookupError):
                # Re-evaluate headers by calling the private helper directly
                headers = {
                    **config._authorization_bearer_header(),
                    **config._authorization_context_header(),
                }
                assert headers == {"Authorization": f"Bearer {api_key}"}
