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

"""Tests for deployment configuration assembly."""

from unittest.mock import MagicMock

import pytest

from datarobot_genai.drmcp.core.dynamic_tools.deployment.config import create_deployment_tool_config
from datarobot_genai.drmcp.core.dynamic_tools.register import ExternalToolRegistrationConfig


@pytest.fixture
def mock_deployment():
    """Create a mock deployment object."""
    deployment = MagicMock()
    deployment.id = "test-deployment-123"
    deployment.label = "Test Weather Tool"
    deployment.description = "A tool for weather forecasting"
    deployment.default_prediction_server = {"url": "https://api.example.com"}
    deployment.serverless_prediction_server = None
    deployment.prediction_environment = None
    return deployment


@pytest.fixture
def mock_metadata():
    """Create mock metadata object."""
    metadata = MagicMock()
    metadata.name = "weather_tool"
    metadata.description = "Get weather forecast"
    metadata.method = "POST"
    metadata.endpoint = "/predict"
    metadata.headers = {"X-Custom-Header": "value"}
    metadata.input_schema = {
        "type": "object",
        "properties": {
            "json": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["location"],
            }
        },
        "required": ["json"],
    }
    return metadata


@pytest.fixture
def mock_api_client():
    """Create a mock API client."""
    client = MagicMock()
    client.token = "test-token"
    client.endpoint = "https://app.datarobot.com/api/v2"
    return client


@pytest.fixture(autouse=True)
def _setup_mocks(monkeypatch, mock_metadata, mock_api_client):
    """Set up common mocks for all tests automatically."""
    monkeypatch.setattr(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.config.get_mcp_tool_metadata",
        lambda *args, **kwargs: mock_metadata,
    )
    monkeypatch.setattr(
        "datarobot_genai.drmcp.core.dynamic_tools.deployment.config.get_api_client",
        lambda *args, **kwargs: mock_api_client,
    )


class TestCreateDeploymentToolConfig:
    """Tests for create_deployment_tool_config function - happy path."""

    @pytest.mark.parametrize(
        ("server_url", "expected_base_url"),
        [
            (
                "https://api.example.com",
                "https://api.example.com/predApi/v1.0/deployments/test-deployment-123/",
            ),
            (
                "https://api.example.com/",
                "https://api.example.com/predApi/v1.0/deployments/test-deployment-123/",
            ),
        ],
    )
    def test_create_config_with_standard_deployment(
        self, mock_deployment, mock_metadata, server_url, expected_base_url
    ):
        """Test creating config from standard deployment handles trailing slashes correctly."""
        mock_deployment.default_prediction_server = {"url": server_url}
        config = create_deployment_tool_config(mock_deployment)

        assert isinstance(config, ExternalToolRegistrationConfig)
        assert (
            config.name,
            config.title,
            config.description,
            config.method,
            config.base_url,
            config.endpoint,
        ) == (
            "test_weather_tool",
            "Test Weather Tool",
            "A tool for weather forecasting",
            "POST",
            expected_base_url,
            "predict",
        )
        assert config.input_schema == mock_metadata.input_schema
        assert config.headers["X-Custom-Header"] == "value"

    @pytest.mark.parametrize(
        ("api_endpoint", "expected_base_url"),
        [
            (
                "https://app.datarobot.com/api/v2",
                "https://app.datarobot.com/api/v2/deployments/test-deployment-123/",
            ),
            (
                "https://app.datarobot.com/api/v2/",
                "https://app.datarobot.com/api/v2/deployments/test-deployment-123/",
            ),
        ],
    )
    def test_create_config_with_serverless_deployment(
        self, mock_deployment, mock_api_client, api_endpoint, expected_base_url
    ):
        """Test creating config from serverless deployment handles trailing slashes correctly."""
        mock_deployment.prediction_environment = {"platform": "datarobotServerless"}
        mock_deployment.serverless_prediction_server = {"url": "https://serverless.example.com"}
        mock_api_client.endpoint = api_endpoint

        config = create_deployment_tool_config(mock_deployment)

        assert config.base_url == expected_base_url
        assert config.headers["Authorization"] == f"Bearer {mock_api_client.token}"

    def test_create_config_merges_headers(self, mock_deployment, mock_api_client):
        """Test that deployment headers are merged with metadata headers."""
        mock_deployment.default_prediction_server = {
            "url": "https://api.example.com",
            "datarobot-key": "test-key-123",
        }
        config = create_deployment_tool_config(mock_deployment)
        assert config.headers == {
            "Authorization": f"Bearer {mock_api_client.token}",
            "datarobot-key": "test-key-123",
            "X-Custom-Header": "value",
        }

    def test_create_config_strips_leading_slash_from_endpoint(self, mock_deployment, mock_metadata):
        """Test that leading slash is removed from endpoint."""
        mock_metadata.endpoint = "/predict/weather"
        config = create_deployment_tool_config(mock_deployment)
        assert config.endpoint == "predict/weather"

    def test_create_config_uses_metadata_description_as_fallback(
        self, mock_deployment, mock_metadata
    ):
        """Test that metadata description is used when deployment has none."""
        mock_deployment.description = None
        config = create_deployment_tool_config(mock_deployment)
        assert config.description == mock_metadata.description

    @pytest.mark.parametrize(
        ("api_endpoint", "expected_base_url"),
        [
            (
                "http://datarobot-nginx/api/v2",
                "http://datarobot-prediction-server:80/predApi/v1.0/deployments/test-deployment-123/",
            ),
            (
                "http://datarobot-nginx/api/v2/",
                "http://datarobot-prediction-server:80/predApi/v1.0/deployments/test-deployment-123/",
            ),
        ],
    )
    def test_create_config_with_onprem_deployment(
        self, mock_deployment, mock_api_client, api_endpoint, expected_base_url
    ):
        """Test creating config for on-prem deployment handles trailing slashes correctly."""
        mock_api_client.endpoint = api_endpoint
        config = create_deployment_tool_config(mock_deployment)
        assert config.base_url == expected_base_url

    def test_create_config_uses_default_prediction_server_key(self, mock_deployment):
        """Test that datarobot-key is used for non-serverless deployments."""
        mock_deployment.default_prediction_server = {
            "url": "https://api.example.com",
            "datarobot-key": "dr-key-12345",
        }
        config = create_deployment_tool_config(mock_deployment)
        assert config.headers["datarobot-key"] == "dr-key-12345"

    @pytest.mark.parametrize(
        ("label", "expected_name"),
        [
            ("Weather Tool", "weather_tool"),
            ("Tool-With-Dashes", "tool_with_dashes"),
            ("Tool [v1]", "tool"),
            ("UPPERCASE TOOL", "uppercase_tool"),
            ("Tool__Multiple__Underscores", "tool_multiple_underscores"),
        ],
    )
    def test_create_config_converts_label_to_valid_name(
        self, mock_deployment, label, expected_name
    ):
        """Test that deployment label is properly converted to valid tool name."""
        mock_deployment.label = label
        config = create_deployment_tool_config(mock_deployment)
        assert config.name == expected_name
