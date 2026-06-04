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

from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from datarobot_genai.drtools.core import credentials


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch):
    """Isolate credential tests from .env file and environment variables."""
    # Disable OpenTelemetry to prevent background export errors
    monkeypatch.setenv("OTEL_ENABLED", "false")

    # Patch model_config to disable .env file loading during tests
    config_without_env = SettingsConfigDict(
        env_file=None,  # Don't load .env file in tests
        case_sensitive=False,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Apply to both credential classes
    monkeypatch.setattr(credentials.DataRobotCredentials, "model_config", config_without_env)
    monkeypatch.setattr(credentials.ToolsAuthCredentials, "model_config", config_without_env)


def test_datarobot_credentials_default_endpoint() -> None:
    """Test DataRobot credentials with default endpoint."""
    env_vars = {
        "DATAROBOT_API_TOKEN": "test-token",
    }
    with patch.dict("os.environ", env_vars, clear=True):  # Clear all env vars
        creds = credentials.DataRobotCredentials()
        assert creds.datarobot_api_token == "test-token"
        assert creds.datarobot_endpoint is not None
        assert creds.datarobot_endpoint.startswith("https://")


def test_datarobot_credentials_custom_endpoint() -> None:
    """Test DataRobot credentials with custom endpoint."""
    env_vars = {
        "DATAROBOT_API_TOKEN": "test-token",
        "DATAROBOT_ENDPOINT": "https://custom.endpoint.com/api/v2",
    }
    with patch.dict("os.environ", env_vars, clear=True):
        creds = credentials.DataRobotCredentials()
        assert creds.datarobot_api_token == "test-token"
        assert creds.datarobot_endpoint == "https://custom.endpoint.com/api/v2"


def test_get_credentials_singleton() -> None:
    """Test get_credentials returns singleton instance."""
    env_vars = {
        "DATAROBOT_API_TOKEN": "test-token",
    }
    with patch.dict("os.environ", env_vars, clear=True):
        # Reset the singleton instance
        credentials._credentials = None

        # First call should create new instance
        creds1 = credentials.get_credentials()
        assert isinstance(creds1, credentials.ToolsAuthCredentials)

        # Second call should return same instance
        creds2 = credentials.get_credentials()
        assert creds2 is creds1


def test_auth_resolution_strategy_defaults_to_http() -> None:
    """Test default auth resolution strategy is http."""
    env_vars = {
        "DATAROBOT_API_TOKEN": "test-token",
    }
    with patch.dict("os.environ", env_vars, clear=True):
        creds = credentials.ToolsAuthCredentials()
        assert creds.auth_resolution_strategy == credentials.AuthResolutionStrategy.HTTP
        assert creds.auth_resolution_strategy == "http"


def test_auth_resolution_strategy_loads_from_env() -> None:
    """Test AUTH_RESOLUTION_STRATEGY env var parses as a string enum."""
    env_vars = {
        "DATAROBOT_API_TOKEN": "test-token",
        "AUTH_RESOLUTION_STRATEGY": "config",
    }
    with patch.dict("os.environ", env_vars, clear=True):
        creds = credentials.ToolsAuthCredentials()
        assert creds.auth_resolution_strategy == credentials.AuthResolutionStrategy.CONFIG
        assert creds.auth_resolution_strategy == "config"


def test_has_datarobot_credentials() -> None:
    """Test ToolsAuthCredentials.has_datarobot_credentials method."""
    # Test with DataRobot credentials
    env_vars = {
        "DATAROBOT_API_TOKEN": "test-token",
    }
    with patch.dict("os.environ", env_vars, clear=True):
        creds = credentials.ToolsAuthCredentials()
        assert creds.has_datarobot_credentials() is True

    # Test without DataRobot credentials
    with patch.dict("os.environ", {}, clear=True):
        try:
            credentials.ToolsAuthCredentials()
            assert False, "Should raise ValidationError"
        except Exception:
            assert True
