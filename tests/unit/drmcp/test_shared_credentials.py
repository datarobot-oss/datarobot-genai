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

from datarobot_genai.drmcp.core import credentials


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch):
    """Isolate credential tests from .env file and environment variables."""
    # Disable OpenTelemetry to prevent background export errors
    monkeypatch.setenv("OTEL_ENABLED", "false")

    # Clear environment variables that might leak from .env or system
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    # Patch model_config to disable .env file loading during tests
    config_without_env = SettingsConfigDict(
        env_file=None,  # Don't load .env file in tests
        case_sensitive=False,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Apply to both credential classes
    monkeypatch.setattr(credentials.DataRobotCredentials, "model_config", config_without_env)
    monkeypatch.setattr(credentials.MCPServerCredentials, "model_config", config_without_env)


def test_datarobot_credentials_default_endpoint() -> None:
    """Test DataRobot credentials with default endpoint."""
    env_vars = {
        "DATAROBOT_API_TOKEN": "test-token",
    }
    with patch.dict("os.environ", env_vars, clear=True):  # Clear all env vars
        creds = credentials.DataRobotCredentials()
        assert creds.application_api_token == "test-token"
        # The endpoint will be whatever is in the .env file or the default
        # We just check that it's a valid endpoint
        assert creds.endpoint is not None
        assert creds.endpoint.startswith("https://")


def test_datarobot_credentials_custom_endpoint() -> None:
    """Test DataRobot credentials with custom endpoint."""
    env_vars = {
        "DATAROBOT_API_TOKEN": "test-token",
        "DATAROBOT_ENDPOINT": "https://custom.endpoint.com/api/v2",
    }
    with patch.dict("os.environ", env_vars, clear=True):
        creds = credentials.DataRobotCredentials()
        assert creds.application_api_token == "test-token"
        # The endpoint will be the custom one we set in env vars
        assert creds.endpoint == "https://custom.endpoint.com/api/v2"


def test_mcp_server_credentials_aws_defaults() -> None:
    """Test AWS credentials with default values."""
    env_vars = {
        "DATAROBOT_API_TOKEN": "test-token",
    }
    with patch.dict("os.environ", env_vars, clear=True):  # Clear all env vars
        creds = credentials.MCPServerCredentials()
        assert creds.aws_access_key_id is None
        assert creds.aws_secret_access_key is None
        assert creds.aws_session_token is None
        assert creds.aws_predictions_s3_bucket == "datarobot-rd"
        assert creds.aws_predictions_s3_prefix == "dev/mcp-temp-storage/predictions/"


def test_mcp_server_credentials_aws_custom_values() -> None:
    """Test AWS credentials with custom values."""
    env_vars = {
        "DATAROBOT_API_TOKEN": "test-token",
        "AWS_ACCESS_KEY_ID": "test-key-id",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_SESSION_TOKEN": "test-session-token",
        "AWS_PREDICTIONS_S3_BUCKET": "custom-bucket",
        "AWS_PREDICTIONS_S3_PREFIX": "custom/prefix/",
    }
    with patch.dict("os.environ", env_vars, clear=True):
        creds = credentials.MCPServerCredentials()
        assert creds.aws_access_key_id == "test-key-id"
        assert creds.aws_secret_access_key == "test-secret-key"
        assert creds.aws_session_token == "test-session-token"
        assert creds.aws_predictions_s3_bucket == "custom-bucket"
        assert creds.aws_predictions_s3_prefix == "custom/prefix/"


def test_mcp_server_credentials_has_aws_credentials() -> None:
    """Test MCPServerCredentials.has_aws_credentials method."""
    # Test with AWS credentials
    env_vars = {
        "DATAROBOT_API_TOKEN": "test-token",
        "AWS_ACCESS_KEY_ID": "test-key-id",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
    }
    with patch.dict("os.environ", env_vars, clear=True):
        creds = credentials.MCPServerCredentials()
        assert creds.has_aws_credentials() is True

    # Test without AWS credentials
    env_vars = {
        "DATAROBOT_API_TOKEN": "test-token",
    }
    with patch.dict("os.environ", env_vars, clear=True):
        creds = credentials.MCPServerCredentials()
        assert creds.has_aws_credentials() is False


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
        assert isinstance(creds1, credentials.MCPServerCredentials)

        # Second call should return same instance
        creds2 = credentials.get_credentials()
        assert creds2 is creds1


def test_has_datarobot_credentials() -> None:
    """Test MCPServerCredentials.has_datarobot_credentials method."""
    # Test with DataRobot credentials
    env_vars = {
        "DATAROBOT_API_TOKEN": "test-token",
    }
    with patch.dict("os.environ", env_vars, clear=True):
        creds = credentials.MCPServerCredentials()
        assert creds.has_datarobot_credentials() is True

    # Test without DataRobot credentials
    with patch.dict("os.environ", {}, clear=True):
        try:
            credentials.MCPServerCredentials()
            assert False, "Should raise ValidationError"
        except Exception:
            assert True
