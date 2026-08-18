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
from http import HTTPStatus
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest
import respx
from datarobot.models.genai.agent.auth import set_authorization_context

from datarobot_genai.core.mcp import MCPConfig
from datarobot_genai.core.mcp.config import clear_workload_endpoint_cache
from datarobot_genai.core.mcp.config import lookup_workload_endpoint

WORKLOAD_ID = "6a6b3d359e6b2c11158c2a13"
API_ENDPOINT = "https://test.datarobot.com/api/v2"
LOOKUP_URL = f"{API_ENDPOINT}/workloads/{WORKLOAD_ID}/"


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_workload_endpoint_cache()
    yield
    clear_workload_endpoint_cache()


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

    def test_is_local_server_true_for_mcp_server_port(self):
        with patch.dict(os.environ, {}, clear=True):
            assert MCPConfig(mcp_server_port=9000).is_local_server is True

    def test_is_local_server_false_for_external_url(self):
        with patch.dict(os.environ, {}, clear=True):
            config = MCPConfig(external_mcp_url="https://mcp.example.com/mcp")
        assert config.is_local_server is False

    def test_is_local_server_false_when_unconfigured(self):
        with patch.dict(os.environ, {}, clear=True):
            assert MCPConfig().is_local_server is False

    def test_is_local_server_false_when_deployment_takes_priority(self):
        # mcp_server_port set but a deployment also configured -> deployment wins.
        with patch.dict(os.environ, {}, clear=True):
            config = MCPConfig(
                mcp_server_port=9000,
                mcp_deployment_id="a" * 24,
                datarobot_endpoint="https://app.datarobot.com",
                datarobot_api_token="tok",
            )
        assert config.is_local_server is False
        assert "directAccess/mcp" in config.server_config["url"]

    def test_is_local_server_false_when_external_takes_priority(self):
        with patch.dict(os.environ, {}, clear=True):
            config = MCPConfig(mcp_server_port=9000, external_mcp_url="https://mcp.example.com/mcp")
        assert config.is_local_server is False
        assert config.server_config["url"] == "https://mcp.example.com/mcp"

    def test_invalid_mcp_server_port_ignored(self):
        # Out-of-range port is dropped (warn + None) rather than producing a config.
        with patch.dict(os.environ, {}, clear=True):
            config = MCPConfig(mcp_server_port=99999)
        assert config.mcp_server_port is None
        assert config.is_local_server is False
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

            # Verify forwarded headers are not included when not provided
            assert "x-datarobot-api-key" not in config.server_config["headers"]

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
        # Verify forwarded headers are not included when not provided
        assert "x-datarobot-api-key" not in config.server_config["headers"]

    @pytest.mark.parametrize(
        "additional_env_params, expected_error_message",
        [
            pytest.param(
                {"DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2"},
                "When using a DataRobot hosted MCP deployment, datarobot_api_token must be set.",
                id="with-endpoint",
            ),
            pytest.param(
                {"DATAROBOT_API_TOKEN": "test-api-key"},
                "When using a DataRobot hosted MCP deployment, datarobot_endpoint must be set.",
                id="with-api-key",
            ),
        ],
    )
    def test_mcp_config_with_datarobot_deployment_id_no_api_key(
        self, additional_env_params, expected_error_message
    ):
        """Test MCP config with DataRobot deployment ID but no API key."""
        deployment_id = "abc123def456789012345678"
        with patch.dict(
            os.environ, {"MCP_DEPLOYMENT_ID": deployment_id, **additional_env_params}, clear=True
        ):
            with pytest.raises(
                ValueError,
                match=expected_error_message,
            ):
                config = MCPConfig()
                config.server_config

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
            # Verify forwarded headers are not included when not provided
            assert "x-datarobot-api-key" not in config.server_config["headers"]

    def test_mcp_config_priority_deployment_over_external(self):
        """Test that MCP_DEPLOYMENT_ID takes priority over EXTERNAL_MCP_URL."""
        external_url = "https://external-mcp.com/mcp"
        deployment_id = "abc123def456789012345678"
        api_key = "test-api-key"
        api_base = "https://app.datarobot.com/api/v2"
        headers = {"X-Custom-Header": "custom-value"}

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": external_url,
                "EXTERNAL_MCP_HEADERS": json.dumps(headers),
                "EXTERNAL_MCP_TRANSPORT": "sse",
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": api_base,
                "DATAROBOT_API_TOKEN": api_key,
            },
            clear=True,
        ):
            config = MCPConfig()
            # Deployment ID takes priority, so it should use deployment config
            expected_url = f"{api_base}/deployments/{deployment_id}/directAccess/mcp"
            assert config.server_config["url"] == expected_url
            assert config.server_config["transport"] == "streamable-http"
            assert "Authorization" in config.server_config["headers"]

    def test_mcp_config_with_external_headers_invalid_json(self):
        """Invalid JSON should return None and log warning, not raise error."""
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
            config = MCPConfig()
            # Invalid JSON should result in None for external_mcp_headers
            assert config.external_mcp_headers is None
            # Server config should still work, just without the invalid headers
            assert config.server_config is not None
            assert config.server_config["headers"] == {}

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
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": api_base,
                "DaTAROBOT_API_TOKEN": api_key,
            },
            clear=True,
        ):
            config = MCPConfig()
            expected_url = f"{api_base}/deployments/{deployment_id}/directAccess/mcp"
            assert config.server_config["url"] == expected_url
            assert config.server_config["headers"]["Authorization"] == f"Bearer {api_key}"
            # Verify forwarded headers are not included when not provided
            assert "x-datarobot-api-key" not in config.server_config["headers"]

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
            # Verify forwarded headers are not included when not provided
            assert "x-datarobot-api-key" not in config.server_config["headers"]

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
            # Verify forwarded headers are not included when not provided
            assert "x-datarobot-api-key" not in config.server_config["headers"]

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
        """Invalid deployment IDs should return None and log warning, not raise error."""
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
            config = MCPConfig()
            # Invalid deployment ID should result in None
            assert config.mcp_deployment_id is None
            # Server config should be None since no valid deployment ID
            assert config.server_config is None
        # This test verifies that invalid deployment IDs are normalized to None
        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": "invalid-format",
                "DATAROBOT_ENDPOINT": "https://app.datarobot.example/api/v2",
                "DATAROBOT_API_TOKEN": "dummy-token",
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.mcp_deployment_id is None
            assert config.server_config is None

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
                # Verify forwarded headers are not included when not provided
                assert "x-datarobot-api-key" not in headers

    def test_mcp_config_with_direct_authorization_context(self, agent_auth_context_data):
        """Test MCPConfig with direct authorization_context parameter."""
        deployment_id = "abc123def456789012345678"
        secret_key = "test-secret-key"

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "SESSION_SECRET_KEY": secret_key,
                "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
                "DATAROBOT_API_TOKEN": "test-api-key",
            },
            clear=True,
        ):
            config = MCPConfig(
                authorization_context=agent_auth_context_data,
            )
            assert config.authorization_context == agent_auth_context_data

            # Verify header is generated correctly
            header = config._authorization_context_header()
            assert "X-DataRobot-Authorization-Context" in header

            # Verify token can be decoded
            token = header["X-DataRobot-Authorization-Context"]
            decoded = config.auth_context_handler.decode(token)
            assert decoded == agent_auth_context_data

            # Verify forwarded headers are not included when not provided
            server_headers = config.server_config["headers"]
            assert "x-datarobot-api-key" not in server_headers

    def test_mcp_config_authorization_context_priority_direct_over_contextvar(
        self, agent_auth_context_data
    ):
        """Test that direct authorization_context param takes priority over ContextVar."""
        deployment_id = "abc123def456789012345678"
        secret_key = "test-secret-key"

        # Set different context in ContextVar
        contextvar_auth = {"user": {"id": "999", "name": "contextvar"}, "identities": []}
        set_authorization_context(contextvar_auth)

        # Create config with explicit authorization_context
        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "SESSION_SECRET_KEY": secret_key,
                "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
                "DATAROBOT_API_TOKEN": "test-api-key",
            },
            clear=True,
        ):
            config = MCPConfig(
                authorization_context=agent_auth_context_data,
            )

            # Verify the direct param is used, not the ContextVar
            header = config._authorization_context_header()
            token = header["X-DataRobot-Authorization-Context"]
            decoded = config.auth_context_handler.decode(token)
            assert decoded == agent_auth_context_data
            assert decoded != contextvar_auth

            # Verify forwarded headers are not included when not provided
            server_headers = config.server_config["headers"]
            assert "x-datarobot-api-key" not in server_headers

    def test_mcp_config_with_empty_authorization_context(self):
        """Test MCPConfig with empty authorization_context dict."""
        deployment_id = "abc123def456789012345678"

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
                "DATAROBOT_API_TOKEN": "test-api-key",
            },
            clear=True,
        ):
            config = MCPConfig(
                authorization_context={},
            )

            # Empty context should not generate a header
            header = config._authorization_context_header()
            assert header == {}

            # Verify forwarded headers are not included when not provided
            server_headers = config.server_config["headers"]
            assert "x-datarobot-api-key" not in server_headers

    def test_mcp_config_with_none_authorization_context(self, agent_auth_context_data):
        """Test MCPConfig with None authorization_context falls back to ContextVar."""
        deployment_id = "abc123def456789012345678"
        secret_key = "test-secret-key"

        # Set context in ContextVar
        set_authorization_context(agent_auth_context_data)

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "SESSION_SECRET_KEY": secret_key,
                "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
                "DATAROBOT_API_TOKEN": "test-api-key",
            },
            clear=True,
        ):
            # Pass None explicitly - should fall back to ContextVar
            config = MCPConfig(
                authorization_context=None,
            )

            # Should fall back to ContextVar
            header = config._authorization_context_header()
            assert "X-DataRobot-Authorization-Context" in header

            token = header["X-DataRobot-Authorization-Context"]
            decoded = config.auth_context_handler.decode(token)
            assert decoded == agent_auth_context_data

            # Verify forwarded headers are not included when not provided
            server_headers = config.server_config["headers"]
            assert "x-datarobot-api-key" not in server_headers

    def test_mcp_config_authorization_context_with_complex_data(self):
        """Test authorization_context with complex nested data structures."""
        deployment_id = "abc123def456789012345678"
        secret_key = "test-secret-key"

        complex_auth_context = {
            "user": {"id": "123", "name": "test", "email": "test@example.com"},
            "identities": [
                {
                    "id": "id123",
                    "type": "user",
                    "provider_type": "github",
                    "provider_user_id": "123",
                    "metadata": {"repos": ["repo1", "repo2"], "stars": 42},
                }
            ],
            "permissions": ["read", "write", "admin"],
            "nested": {"level1": {"level2": {"level3": "deep value"}}},
        }

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "SESSION_SECRET_KEY": secret_key,
                "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
                "DATAROBOT_API_TOKEN": "test-api-key",
            },
            clear=True,
        ):
            config = MCPConfig(
                authorization_context=complex_auth_context,
            )

            header = config._authorization_context_header()
            token = header["X-DataRobot-Authorization-Context"]
            decoded = config.auth_context_handler.decode(token)

            # Verify all nested data is preserved
            assert decoded == complex_auth_context
            assert decoded["nested"]["level1"]["level2"]["level3"] == "deep value"

            # Verify forwarded headers are not included when not provided
            server_headers = config.server_config["headers"]
            assert "x-datarobot-api-key" not in server_headers

    def test_mcp_config_authorization_context_with_external_mcp(self, agent_auth_context_data):
        """Test that authorization_context is stored but not used for external MCP."""
        test_url = "https://external-mcp.example.com/mcp"
        secret_key = "test-secret-key"

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
                "SESSION_SECRET_KEY": secret_key,
            },
            clear=True,
        ):
            config = MCPConfig(authorization_context=agent_auth_context_data)

            # Config should store the context
            assert config.authorization_context == agent_auth_context_data

            # But server config should not include the auth context header for external MCP
            assert "X-DataRobot-Authorization-Context" not in config.server_config["headers"]

    def test_mcp_config_authorization_context_roundtrip(self, agent_auth_context_data):
        """Test full encode-decode roundtrip of authorization_context."""
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "fake_api_key"
        secret_key = "test-secret-key"

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "SESSION_SECRET_KEY": secret_key,
                "DATAROBOT_API_TOKEN": api_key,
                "DATAROBOT_ENDPOINT": api_base,
            },
            clear=True,
        ):
            # Create config with auth context
            config1 = MCPConfig(
                authorization_context=agent_auth_context_data,
            )

            # Get the header with JWT token
            headers = config1.server_config["headers"]
            jwt_token = headers["X-DataRobot-Authorization-Context"]

            # Create a new config and decode the token
            config2 = MCPConfig(api_base=api_base, api_key=api_key)
            decoded_context = config2.auth_context_handler.decode(jwt_token)

            # Verify roundtrip preserves all data
            assert decoded_context == agent_auth_context_data

            # Verify forwarded headers are not included when not provided
            headers1 = config1.server_config["headers"]
            assert "x-datarobot-api-key" not in headers1

    def test_mcp_config_authorization_context_with_missing_secret_key(
        self, agent_auth_context_data
    ):
        """Test authorization_context encoding with missing secret key shows warning."""
        deployment_id = "abc123def456789012345678"

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
                "DATAROBOT_API_TOKEN": "test-api-key",
            },
            clear=True,
        ):
            with pytest.warns(UserWarning, match="No secret key provided"):
                config = MCPConfig(
                    authorization_context=agent_auth_context_data,
                )

                # Should still generate a header, but with empty key (insecure)
                header = config._authorization_context_header()
                assert "X-DataRobot-Authorization-Context" in header

                # Verify forwarded headers are not included when not provided
                server_headers = config.server_config["headers"]
                assert "x-datarobot-api-key" not in server_headers

    def test_mcp_config_with_forwarded_headers(self, agent_auth_context_data):
        """Test MCPConfig with forwarded headers including scoped token."""
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "test-api-key"
        secret_key = "my-secret-key"
        forwarded_headers = {
            "x-datarobot-api-key": "scoped-token-123",
            "x-custom-header": "custom-value",
        }

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
            config = MCPConfig(forwarded_headers=forwarded_headers)
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Verify forwarded headers are included
            assert headers["x-datarobot-api-key"] == "scoped-token-123"
            assert headers["x-custom-header"] == "custom-value"
            # Verify other headers are still present
            assert headers["Authorization"] == f"Bearer {api_key}"
            assert "X-DataRobot-Authorization-Context" in headers

    def test_mcp_config_with_forwarded_headers_none(self):
        """Test MCPConfig with None forwarded headers doesn't include them."""
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
            config = MCPConfig(forwarded_headers=None)
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Verify forwarded headers are not included
            assert "x-datarobot-api-key" not in headers
            # Verify other headers are still present
            assert headers["Authorization"] == f"Bearer {api_key}"

    def test_mcp_config_with_forwarded_headers_empty_dict(self):
        """Test MCPConfig with empty forwarded headers dict."""
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
            config = MCPConfig(forwarded_headers={})
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Verify forwarded headers are not included (empty dict)
            assert "x-datarobot-api-key" not in headers
            # Verify other headers are still present
            assert headers["Authorization"] == f"Bearer {api_key}"

    def test_mcp_config_with_forwarded_headers_scoped_token_only(self):
        """Test MCPConfig with only scoped token in forwarded headers."""
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        forwarded_headers = {"x-datarobot-api-key": "scoped-token-456"}

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": api_base,
                "DATAROBOT_API_TOKEN": "test-api-key",
            },
            clear=True,
        ):
            config = MCPConfig(forwarded_headers=forwarded_headers)
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Verify scoped token is included
            assert headers["x-datarobot-api-key"] == "scoped-token-456"
            # Verify Authorization header is present (from DATAROBOT_API_TOKEN)
            assert headers["Authorization"] == "Bearer test-api-key"

    def test_mcp_config_forwarded_headers_protected_authorization_header_not_overwritten(self):
        """Test that authorization header from MCPConfig overwrite any in forwarded_headers."""
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "correct-api-key"
        forwarded_headers = {
            "x-datarobot-api-key": "scoped-token-789",
            "Authorization": "Bearer wrong-token",  # Will be overwritten by MCPConfig
            "authorization": "Bearer another-wrong-token",  # Will be overwritten by MCPConfig
        }

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": api_base,
                "DATAROBOT_API_TOKEN": api_key,
            },
            clear=True,
        ):
            config = MCPConfig(forwarded_headers=forwarded_headers)
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Verify protected authorization header from MCPConfig overwrite forwarded_headers
            assert headers["Authorization"] == f"Bearer {api_key}"
            # Verify forwarded headers that are not authorization are included
            assert headers["x-datarobot-api-key"] == "scoped-token-789"

    def test_mcp_config_external_url_with_localhost_no_forwarded_headers(self):
        """Test that forwarded headers are NOT included for external MCP URLs (even localhost)."""
        test_url = "http://localhost:8080/mcp"
        forwarded_headers = {
            "x-datarobot-api-key": "scoped-token-123",
            "x-custom-header": "custom-value",
        }

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
            },
            clear=True,
        ):
            config = MCPConfig(forwarded_headers=forwarded_headers)
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Verify forwarded headers are NOT included for external URLs
            assert "x-datarobot-api-key" not in headers
            assert "x-custom-header" not in headers
            assert headers == {}

    def test_mcp_config_external_url_with_127_0_0_1_no_forwarded_headers(self):
        """Test that forwarded headers are NOT included for external MCP URLs (even 127.0.0.1)."""
        test_url = "http://127.0.0.1:8080/mcp"
        forwarded_headers = {
            "x-datarobot-api-key": "scoped-token-456",
            "x-test-header": "test-value",
        }

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
            },
            clear=True,
        ):
            config = MCPConfig(forwarded_headers=forwarded_headers)
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Verify forwarded headers are NOT included for external URLs
            assert "x-datarobot-api-key" not in headers
            assert "x-test-header" not in headers
            assert headers == {}

    def test_mcp_config_external_url_non_localhost_no_forwarded_headers(self):
        """Test that forwarded headers are NOT included for non-localhost external MCP URLs."""
        test_url = "https://external-mcp.example.com/mcp"
        forwarded_headers = {
            "x-datarobot-api-key": "scoped-token-789",
            "x-custom-header": "should-not-appear",
        }

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
            },
            clear=True,
        ):
            config = MCPConfig(forwarded_headers=forwarded_headers)
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Verify forwarded headers are NOT included for non-localhost
            assert "x-datarobot-api-key" not in headers
            assert "x-custom-header" not in headers
            assert headers == {}

    def test_mcp_config_external_url_localhost_with_external_headers_only(self):
        """Test that forwarded headers are not merged with external headers for external URLs."""
        test_url = "http://localhost:8080/mcp"
        forwarded_headers = {
            "x-datarobot-api-key": "scoped-token-123",
            "x-forwarded-header": "forwarded-value",
        }
        external_headers = {
            "X-Custom-Header": "external-value",
            "X-Another-Header": "another-value",
        }

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
                "EXTERNAL_MCP_HEADERS": json.dumps(external_headers),
            },
            clear=True,
        ):
            config = MCPConfig(forwarded_headers=forwarded_headers)
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Verify only external headers are present (forwarded headers not included)
            assert "x-datarobot-api-key" not in headers
            assert "x-forwarded-header" not in headers
            assert headers["X-Custom-Header"] == "external-value"
            assert headers["X-Another-Header"] == "another-value"

    def test_mcp_config_external_url_localhost_external_headers_only(self):
        """Test that forwarded headers are not included for external URLs."""
        test_url = "http://localhost:8080/mcp"
        forwarded_headers = {
            "x-datarobot-api-key": "forwarded-token",
            "X-Custom-Header": "forwarded-value",
        }
        external_headers = {
            "X-Custom-Header": "external-value",
            "X-Another-Header": "another-value",
        }

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
                "EXTERNAL_MCP_HEADERS": json.dumps(external_headers),
            },
            clear=True,
        ):
            config = MCPConfig(forwarded_headers=forwarded_headers)
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Verify forwarded headers are NOT present
            assert "x-datarobot-api-key" not in headers
            # Verify only external headers are present
            assert headers["X-Custom-Header"] == "external-value"
            assert headers["X-Another-Header"] == "another-value"

    def test_mcp_config_external_url_with_external_headers_only(self):
        """Test external MCP URL with only external headers (no forwarded headers)."""
        test_url = "https://mcp-server.example.com/mcp"
        external_headers = {
            "X-Custom-Header": "custom-value",
            "X-Auth-Token": "auth-token-123",
        }

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
                "EXTERNAL_MCP_HEADERS": json.dumps(external_headers),
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Verify external headers are present
            assert headers["X-Custom-Header"] == "custom-value"
            assert headers["X-Auth-Token"] == "auth-token-123"

    def test_mcp_config_external_url_localhost_no_forwarded_headers(self):
        """Test localhost external MCP URL without forwarded headers."""
        test_url = "http://localhost:8080/mcp"

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Verify headers are empty when no forwarded headers provided
            assert headers == {}

    def test_mcp_config_external_url_127_0_0_1_with_external_headers_only(self):
        """Test 127.0.0.1 external MCP URL with only external headers (forwarded not included)."""
        test_url = "http://127.0.0.1:3000/mcp"
        forwarded_headers = {"x-forwarded": "value"}
        external_headers = {"X-External": "external-value"}

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
                "EXTERNAL_MCP_HEADERS": json.dumps(external_headers),
            },
            clear=True,
        ):
            config = MCPConfig(forwarded_headers=forwarded_headers)
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Verify only external headers are present (forwarded not included)
            assert "x-forwarded" not in headers
            assert headers["X-External"] == "external-value"

    def test_mcp_config_external_url_localhost_in_domain(self):
        """Test that localhost detection works when localhost appears in domain name."""
        test_url = "https://mylocalhost.example.com/mcp"
        forwarded_headers = {"x-forwarded": "value"}

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
            },
            clear=True,
        ):
            config = MCPConfig(forwarded_headers=forwarded_headers)
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Should NOT forward headers (localhost is in domain, not hostname)
            assert "x-forwarded" not in headers

    def test_mcp_config_external_url_127_0_0_1_in_path(self):
        """Test that 127.0.0.1 detection works correctly (not in path)."""
        test_url = "https://example.com/127.0.0.1/mcp"
        forwarded_headers = {"x-forwarded": "value"}

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": test_url,
            },
            clear=True,
        ):
            config = MCPConfig(forwarded_headers=forwarded_headers)
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Should NOT forward headers (127.0.0.1 is in path, not hostname)
            assert "x-forwarded" not in headers

    def test_mcp_config_deployment_id_with_forwarded_headers(self):
        """Test that forwarded headers are included for DataRobot deployment ID config."""
        deployment_id = "abc123def456789012345678"
        api_base = "https://app.datarobot.com/api/v2"
        api_key = "test-api-key"
        forwarded_headers = {
            "x-datarobot-api-key": "scoped-token-999",
            "x-custom-header": "custom-value",
        }

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "DATAROBOT_ENDPOINT": api_base,
                "DATAROBOT_API_TOKEN": api_key,
            },
            clear=True,
        ):
            config = MCPConfig(forwarded_headers=forwarded_headers)
            assert config.server_config is not None
            headers = config.server_config["headers"]

            # Verify forwarded headers are included
            assert headers["x-datarobot-api-key"] == "scoped-token-999"
            assert headers["x-custom-header"] == "custom-value"
            # Verify auth headers are also present
            assert headers["Authorization"] == f"Bearer {api_key}"

    def test_mcp_config_localhost_server_running(self):
        """Test MCP config with localhost server port when server is running."""
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.json.return_value = {"message": "DataRobot MCP Server is running"}

        with patch.dict(
            os.environ,
            {
                "DATAROBOT_API_TOKEN": "test-api-key",
            },
            clear=True,
        ):
            config = MCPConfig(mcp_server_port=8080)
            assert config.server_config is not None
            assert config.server_config["url"] == "http://localhost:8080/mcp"
            assert config.server_config["transport"] == "streamable-http"
            assert "Authorization" in config.server_config["headers"]
            assert config.server_config["headers"]["Authorization"] == "Bearer test-api-key"

    def test_mcp_config_build_authenticated_headers(self):
        """Test _build_authenticated_headers method."""
        api_key = "test-api-key"
        forwarded_headers = {"x-custom": "value"}

        with patch.dict(
            os.environ,
            {
                "DATAROBOT_API_TOKEN": api_key,
            },
            clear=True,
        ):
            config = MCPConfig(forwarded_headers=forwarded_headers)
            headers = config._build_authenticated_headers()

            assert headers["x-custom"] == "value"
            assert headers["Authorization"] == f"Bearer {api_key}"
            # Authorization context header may or may not be present depending on test setup
            assert (
                "X-DataRobot-Authorization-Context" in headers
                or "X-DataRobot-Authorization-Context" not in headers
            )

    def test_mcp_config_priority_deployment_external_localhost(self):
        """Test priority: deployment > external > localhost."""
        deployment_id = "abc123def456789012345678"
        external_url = "https://external-mcp.com/mcp"
        api_key = "test-api-key"
        api_base = "https://app.datarobot.com/api/v2"

        with patch.dict(
            os.environ,
            {
                "MCP_DEPLOYMENT_ID": deployment_id,
                "EXTERNAL_MCP_URL": external_url,
                "DATAROBOT_ENDPOINT": api_base,
                "DATAROBOT_API_TOKEN": api_key,
            },
            clear=True,
        ):
            config = MCPConfig(mcp_server_port=8080)
            # Deployment should take priority
            expected_url = f"{api_base}/deployments/{deployment_id}/directAccess/mcp"
            assert config.server_config["url"] == expected_url

    def test_mcp_config_priority_external_over_localhost(self):
        """Test priority: external > localhost when no deployment."""
        external_url = "https://external-mcp.com/mcp"

        with patch.dict(
            os.environ,
            {
                "EXTERNAL_MCP_URL": external_url,
            },
            clear=True,
        ):
            config = MCPConfig(mcp_server_port=8080)
            # External should take priority over localhost
            assert config.server_config["url"] == external_url

    # ------------------------------------------------------------------
    # Workload MCP mode (mcp_workload_id)
    # ------------------------------------------------------------------

    @pytest.fixture
    def unresolvable_workload_endpoint(self):
        """Make the Workload API lookup answer nothing.

        Also keeps these tests off the network: without a stub the lookup would
        really call ``{DATAROBOT_ENDPOINT}/api/v2/workloads/<id>/``.
        """
        with patch(
            "datarobot_genai.core.mcp.config.lookup_workload_endpoint", return_value=None
        ) as lookup:
            yield lookup

    @pytest.fixture
    def workload_endpoint(self):
        """Report a workload served from a per-enclave host, with no /api/v2 prefix."""
        with patch(
            "datarobot_genai.core.mcp.config.lookup_workload_endpoint",
            return_value=f"https://test.datarobot.com/workloads/{WORKLOAD_ID}/",
        ) as lookup:
            yield lookup

    def test_mcp_config_with_workload_id(self, agent_auth_context_data, workload_endpoint):
        """GIVEN a cluster that serves workloads through the inference endpoint.

        WHEN the MCP server is addressed by workload ID,
        THEN the URL is the endpoint the platform reports plus ``/mcp`` and the credentials
        match custom-model deployment mode.
        """
        workload_id = "6a6b3d359e6b2c11158c2a13"
        api_base = "https://test.datarobot.com"
        api_key = "test-api-key"

        set_authorization_context(agent_auth_context_data)

        with patch.dict(
            os.environ,
            {
                "MCP_WORKLOAD_ID": workload_id,
                "DATAROBOT_ENDPOINT": api_base,
                "DATAROBOT_API_TOKEN": api_key,
                "SESSION_SECRET_KEY": "my-secret-key",
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.mcp_workload_id == workload_id
            assert config.server_config is not None
            assert config.server_config["url"] == (
                f"https://test.datarobot.com/workloads/{workload_id}/mcp"
            )
            assert config.server_config["transport"] == "streamable-http"
            assert config.server_config["headers"]["Authorization"] == f"Bearer {api_key}"

            # Auth-context header propagates the same way as deployment mode.
            jwt_token = config.server_config["headers"]["X-DataRobot-Authorization-Context"]
            decoded = config.auth_context_handler.decode(jwt_token)
            assert decoded == agent_auth_context_data

        # The lookup is asked at the API host, which is not the workload's host.
        assert workload_endpoint.call_args.args == (workload_id,)
        assert workload_endpoint.call_args.kwargs["endpoint"] == api_base
        assert workload_endpoint.call_args.kwargs["token"] == api_key

    def test_unresolvable_workload_means_no_mcp_server(self, unresolvable_workload_endpoint):
        """GIVEN a workload whose endpoint the platform will not report."""
        with patch.dict(
            os.environ,
            {
                "MCP_WORKLOAD_ID": "a" * 24,
                "DATAROBOT_ENDPOINT": "https://app.datarobot.com",
                "DATAROBOT_API_TOKEN": "tok",
            },
            clear=True,
        ):
            assert MCPConfig().server_config is None

    @pytest.mark.parametrize(
        "other_env",
        [
            pytest.param({"MCP_DEPLOYMENT_ID": "c" * 24}, id="deployment"),
            pytest.param({"EXTERNAL_MCP_URL": "https://external.example/mcp"}, id="external"),
            pytest.param({"MCP_SERVER_PORT": "9000"}, id="local"),
        ],
    )
    def test_workload_takes_precedence_over_other_addresses(self, other_env, workload_endpoint):
        """GIVEN a workload ID alongside another address source.

        One set of environment variables resolves one server, so precedence picks —
        it does not raise. Co-occurrence is usually intentional: a developer's .env
        keeps its own address while infra injects the deployed one as a runtime
        parameter, which is what lets one .env work on a laptop and in production.
        """
        with patch.dict(
            os.environ,
            {
                "MCP_WORKLOAD_ID": WORKLOAD_ID,
                "DATAROBOT_ENDPOINT": "https://app.datarobot.com",
                "DATAROBOT_API_TOKEN": "tok",
                **other_env,
            },
            clear=True,
        ):
            assert (
                MCPConfig().server_config["url"]
                == f"https://test.datarobot.com/workloads/{WORKLOAD_ID}/mcp"
            )

    def test_invalid_workload_id_normalized_to_none(self):
        """Invalid workload IDs should return None and log a warning (like deployment_id)."""
        with patch.dict(
            os.environ,
            {
                "MCP_WORKLOAD_ID": "short-id",
                "DATAROBOT_ENDPOINT": "https://app.datarobot.com",
                "DATAROBOT_API_TOKEN": "tok",
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.mcp_workload_id is None
            # With no valid workload / deployment / external / local, nothing to configure.
            assert config.server_config is None

    def test_workload_id_whitespace_trim(self):
        """Whitespace around a valid workload ID is stripped."""
        workload_id = "a" * 24
        with patch.dict(
            os.environ,
            {
                "MCP_WORKLOAD_ID": f"  {workload_id}  ",
                "DATAROBOT_ENDPOINT": "https://app.datarobot.com",
                "DATAROBOT_API_TOKEN": "tok",
            },
            clear=True,
        ):
            config = MCPConfig()
            assert config.mcp_workload_id == workload_id

    @pytest.mark.parametrize(
        "missing_env, expected_error_message",
        [
            pytest.param(
                {"DATAROBOT_ENDPOINT": "https://app.datarobot.com"},
                "When using a DataRobot workload MCP, datarobot_api_token must be set.",
                id="with-endpoint",
            ),
            pytest.param(
                {"DATAROBOT_API_TOKEN": "tok"},
                "When using a DataRobot workload MCP, datarobot_endpoint must be set.",
                id="with-api-key",
            ),
        ],
    )
    def test_workload_missing_endpoint_or_token_raises(self, missing_env, expected_error_message):
        with patch.dict(
            os.environ,
            {
                "MCP_WORKLOAD_ID": "a" * 24,
                **missing_env,
            },
            clear=True,
        ):
            config = MCPConfig()
            with pytest.raises(ValueError, match=expected_error_message):
                config.server_config

    def test_workload_with_forwarded_headers(self, workload_endpoint):
        """Forwarded headers propagate in workload mode (same as deployment mode)."""
        forwarded_headers = {
            "x-datarobot-api-key": "scoped-token-42",
            "x-custom-header": "custom-value",
        }
        with patch.dict(
            os.environ,
            {
                "MCP_WORKLOAD_ID": "a" * 24,
                "DATAROBOT_ENDPOINT": "https://app.datarobot.com",
                "DATAROBOT_API_TOKEN": "tok",
            },
            clear=True,
        ):
            config = MCPConfig(forwarded_headers=forwarded_headers)
            headers = config.server_config["headers"]
            assert headers["x-datarobot-api-key"] == "scoped-token-42"
            assert headers["x-custom-header"] == "custom-value"
            assert headers["Authorization"] == "Bearer tok"


class TestLookupWorkloadEndpoint:
    @respx.mock
    def test_returns_the_endpoint_the_platform_reports(self):
        # GIVEN a cluster that serves workloads, so the
        # workload's host differs from the API host
        route = respx.get(LOOKUP_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "running",
                    "endpoint": "https://test.datarobot.com/workloads/{WORKLOAD_ID}/",
                },
            )
        )
        # WHEN the endpoint is looked up
        resolved = lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok")
        # THEN the platform's answer is used verbatim, host and prefix included
        assert resolved == "https://test.datarobot.com/workloads/{WORKLOAD_ID}/"
        assert route.called

    @respx.mock
    def test_endpoint_is_normalized_before_the_lookup(self):
        # GIVEN DATAROBOT_ENDPOINT spelled without /api/v2
        route = respx.get(LOOKUP_URL).mock(
            return_value=httpx.Response(
                200, json={"endpoint": "https://test.datarobot.com/workloads/{WORKLOAD_ID}/"}
            )
        )
        # WHEN the endpoint is looked up
        lookup_workload_endpoint(
            WORKLOAD_ID,
            endpoint="https://test.datarobot.com/",
            token="tok",
        )
        # THEN the request still goes to /api/v2/workloads/<id>/
        assert route.called

    @respx.mock
    def test_bearer_token_is_sent_once(self):
        # GIVEN a token that already carries the Bearer prefix
        route = respx.get(LOOKUP_URL).mock(
            return_value=httpx.Response(
                200, json={"endpoint": "https://test.datarobot.com/workloads/{WORKLOAD_ID}/"}
            )
        )
        # WHEN the endpoint is looked up
        lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="Bearer tok")
        # THEN the prefix is not doubled
        assert route.calls.last.request.headers["Authorization"] == "Bearer tok"

    @respx.mock
    def test_running_workloads_answer_is_cached(self):
        # GIVEN a running workload whose endpoint has been resolved once
        route = respx.get(LOOKUP_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "running",
                    "endpoint": "https://test.datarobot.com/workloads/{WORKLOAD_ID}/",
                },
            )
        )
        lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok")
        # WHEN it is looked up again
        second_lookup = lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok")
        # THEN the platform is asked only once — a running workload's route is settled
        assert second_lookup == "https://test.datarobot.com/workloads/{WORKLOAD_ID}/"
        assert route.call_count == 1

    @pytest.mark.parametrize("status", ["submitted", "provisioning", "launching", "suspended"])
    @respx.mock
    def test_a_workload_that_is_not_running_is_not_cached(self, status):
        """GIVEN a workload that has not been scheduled yet.

        On a cluster that advertises the Covalent-reported inference endpoint, the
        API answers with the prediction-gateway URL until the workload is scheduled
        — the wrong route there. Remembering it would pin the agent to it for the
        life of the process, so the answer is used but not cached.
        """
        gateway_url = f"https://app.datarobot.com/api/v2/endpoints/workloads/{WORKLOAD_ID}/"
        route = respx.get(LOOKUP_URL).mock(
            side_effect=[
                httpx.Response(200, json={"status": status, "endpoint": gateway_url}),
                httpx.Response(
                    200,
                    json={
                        "status": "running",
                        "endpoint": "https://test.datarobot.com/workloads/{WORKLOAD_ID}/",
                    },
                ),
            ]
        )
        # WHEN it is looked up while starting, and again once it is running
        assert lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok") == (
            gateway_url
        )
        assert lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok") == (
            "https://test.datarobot.com/workloads/{WORKLOAD_ID}/"
        )
        # THEN the stale answer was re-resolved rather than kept
        assert route.call_count == 2

    @respx.mock
    def test_a_different_api_endpoint_is_looked_up_separately(self):
        # GIVEN the same workload ID resolved against two clusters
        respx.get(LOOKUP_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "running",
                    "endpoint": f"https://test.datarobot.com/workloads/{WORKLOAD_ID}/",
                },
            )
        )
        other = respx.get(f"https://other.datarobot.com/api/v2/workloads/{WORKLOAD_ID}/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "running",
                    "endpoint": (
                        f"https://other.datarobot.com/api/v2/endpoints/workloads/{WORKLOAD_ID}/"
                    ),
                },
            )
        )
        lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok")
        # WHEN the second cluster is asked
        resolved = lookup_workload_endpoint(
            WORKLOAD_ID, endpoint="https://other.datarobot.com/api/v2", token="tok"
        )
        # THEN the cache does not leak one cluster's answer into the other
        assert other.called
        assert resolved == f"https://other.datarobot.com/api/v2/endpoints/workloads/{WORKLOAD_ID}/"

    @pytest.mark.parametrize(
        "response",
        [
            pytest.param(httpx.Response(403, json={"message": "no permission"}), id="forbidden"),
            pytest.param(httpx.Response(404, json={"message": "not found"}), id="not-found"),
            pytest.param(httpx.Response(500, text="boom"), id="server-error"),
            pytest.param(httpx.Response(200, text="not json"), id="non-json-body"),
        ],
    )
    @respx.mock
    def test_unreadable_workload_yields_no_answer(self, response, caplog):
        # GIVEN a lookup the platform will not answer
        respx.get(LOOKUP_URL).mock(return_value=response)
        # WHEN the endpoint is looked up
        with caplog.at_level("WARNING"):
            resolved = lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok")
        # THEN the caller is told nothing was resolved, and what to do about it
        assert resolved is None
        assert "may read the workload" in caplog.text

    @respx.mock
    def test_transport_error_yields_no_answer(self, caplog):
        # GIVEN an unreachable API host
        respx.get(LOOKUP_URL).mock(side_effect=httpx.ConnectError("unreachable"))
        # WHEN the endpoint is looked up
        with caplog.at_level("WARNING"):
            assert lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok") is None
        assert "Could not read the endpoint of workload" in caplog.text

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"status": "stopped"}, id="missing"),
            pytest.param({"status": "stopped", "endpoint": None}, id="null"),
            pytest.param({"status": "stopped", "endpoint": "   "}, id="blank"),
        ],
    )
    @respx.mock
    def test_workload_without_an_endpoint_yields_no_answer(self, payload, caplog):
        # GIVEN a workload that is not serving yet, so it has no endpoint
        respx.get(LOOKUP_URL).mock(return_value=httpx.Response(200, json=payload))
        # WHEN the endpoint is looked up
        with caplog.at_level("WARNING"):
            resolved = lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok")
        # THEN nothing is invented, and the log says it may not be running
        assert resolved is None
        assert "may not be running yet" in caplog.text

    @respx.mock
    def test_a_failed_lookup_is_not_cached(self):
        # GIVEN a lookup that fails once and then succeeds
        route = respx.get(LOOKUP_URL).mock(
            side_effect=[
                httpx.Response(503, text="unavailable"),
                httpx.Response(
                    200, json={"endpoint": "https://test.datarobot.com/workloads/{WORKLOAD_ID}/"}
                ),
            ]
        )
        assert lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok") is None
        # WHEN it is retried
        resolved = lookup_workload_endpoint(WORKLOAD_ID, endpoint=API_ENDPOINT, token="tok")
        # THEN the transient failure was not remembered
        assert resolved == "https://test.datarobot.com/workloads/{WORKLOAD_ID}/"
        assert route.call_count == 2
