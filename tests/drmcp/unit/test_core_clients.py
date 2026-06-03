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
from collections.abc import Iterator
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from datarobot.auth.identity import Identity
from datarobot.auth.session import AuthCtx
from datarobot.auth.users import User

from datarobot_genai.drmcp.core.clients import dr
from datarobot_genai.drmcp.core.clients import (
    setup_and_return_dr_api_client_with_static_config_in_container,
)
from datarobot_genai.drmcp.core.routes_utils import prefix_mount_path
from datarobot_genai.drtools.core.auth import _extract_token_from_auth_context
from datarobot_genai.drtools.core.auth import _extract_token_from_headers
from datarobot_genai.drtools.core.auth import _extract_token_from_headers_with_fallback


@pytest.fixture
def mock_module_under_test() -> str:
    return "datarobot_genai.drmcp.core.clients"


class TestDRAPIClientWithStaticConfigInContainer:
    @pytest.fixture
    def mock_get_credentials(self, mock_module_under_test: str) -> Iterator[Mock]:
        with patch(f"{mock_module_under_test}.get_credentials") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_dr_client_cls(self) -> Iterator[Mock]:
        with patch.object(dr, "Client") as mock_cls:
            yield mock_cls

    def test_setup_and_return_dr_api_client_with_static_config_in_container(
        self,
        mock_dr_client_cls: Mock,
        mock_get_credentials: Mock,
    ) -> None:
        output = setup_and_return_dr_api_client_with_static_config_in_container()

        mock_get_credentials.assert_called_once_with()
        mock_credentials = mock_get_credentials.return_value
        mock_dr_client_cls.assert_called_once_with(
            token=mock_credentials.datarobot.application_api_token,
            endpoint=mock_credentials.datarobot.endpoint,
        )
        assert output == mock_dr_client_cls.return_value


class TestPrefixMountPath:
    """Test cases for prefix_mount_path function."""

    @patch("datarobot_genai.drmcp.core.routes_utils.get_config")
    def test_mount_path_root(self, mock_get_config):
        """Test with mount_path as root."""
        mock_config = Mock()
        mock_config.mount_path = "/"
        mock_get_config.return_value = mock_config

        result = prefix_mount_path("/test/endpoint")
        assert result == "/test/endpoint"

    @patch("datarobot_genai.drmcp.core.routes_utils.get_config")
    def test_mount_path_with_trailing_slash(self, mock_get_config):
        """Test with mount_path ending with slash."""
        mock_config = Mock()
        mock_config.mount_path = "/api/"
        mock_get_config.return_value = mock_config

        result = prefix_mount_path("/test/endpoint")
        assert result == "/api/test/endpoint"

    @patch("datarobot_genai.drmcp.core.routes_utils.get_config")
    def test_mount_path_without_trailing_slash(self, mock_get_config):
        """Test with mount_path not ending with slash."""
        mock_config = Mock()
        mock_config.mount_path = "/api"
        mock_get_config.return_value = mock_config

        result = prefix_mount_path("/test/endpoint")
        assert result == "/api/test/endpoint"

    @patch("datarobot_genai.drmcp.core.routes_utils.get_config")
    def test_endpoint_without_leading_slash(self, mock_get_config):
        """Test with endpoint not starting with slash."""
        mock_config = Mock()
        mock_config.mount_path = "/api"
        mock_get_config.return_value = mock_config

        result = prefix_mount_path("test/endpoint")
        assert result == "/api/test/endpoint"

    @patch("datarobot_genai.drmcp.core.routes_utils.get_config")
    def test_endpoint_with_leading_slash(self, mock_get_config):
        """Test with endpoint starting with slash."""
        mock_config = Mock()
        mock_config.mount_path = "/api"
        mock_get_config.return_value = mock_config

        result = prefix_mount_path("/test/endpoint")
        assert result == "/api/test/endpoint"

    @patch("datarobot_genai.drmcp.core.routes_utils.get_config")
    def test_empty_endpoint(self, mock_get_config):
        """Test with empty endpoint."""
        mock_config = Mock()
        mock_config.mount_path = "/api"
        mock_get_config.return_value = mock_config

        result = prefix_mount_path("")
        assert result == "/api/"

    @patch("datarobot_genai.drmcp.core.routes_utils.get_config")
    def test_root_mount_path_with_empty_endpoint(self, mock_get_config):
        """Test with root mount_path and empty endpoint."""
        mock_config = Mock()
        mock_config.mount_path = "/"
        mock_get_config.return_value = mock_config

        result = prefix_mount_path("")
        assert result == ""


class TestExtractTokenFromHeaders:
    """Test cases for _extract_token_from_headers function."""

    def test_extract_bearer_token_from_authorization_header(self):
        """Test extracting Bearer token from authorization header."""
        headers = {"authorization": "Bearer test-token-123"}
        result = _extract_token_from_headers(headers)
        assert result == "test-token-123"

    def test_extract_bearer_token_case_insensitive(self):
        """Test that Bearer prefix is case-insensitive."""
        headers = {"authorization": "BEARER test-token-123"}
        result = _extract_token_from_headers(headers)
        assert result == "test-token-123"

    def test_extract_plain_token_from_authorization_header(self):
        """Test extracting plain token (without Bearer prefix) from authorization header."""
        headers = {"authorization": "plain-token-123"}
        result = _extract_token_from_headers(headers)
        assert result == "plain-token-123"

    def test_extract_token_from_x_datarobot_api_token_header(self):
        """Test extracting token from x-datarobot-api-token header."""
        headers = {"x-datarobot-api-token": "Bearer api-token-456"}
        result = _extract_token_from_headers(headers)
        assert result == "api-token-456"

    def test_extract_token_from_x_datarobot_api_key_header(self):
        """Test extracting token from x-datarobot-api-key header."""
        headers = {"x-datarobot-api-key": "Bearer api-key-789"}
        result = _extract_token_from_headers(headers)
        assert result == "api-key-789"

    def test_prefers_api_key_over_authorization(self):
        """Test current candidate order where API key headers are checked before authorization."""
        headers = {
            "authorization": "Bearer auth-token",
            "x-datarobot-api-token": "Bearer api-token",
            "x-datarobot-api-key": "Bearer api-key",
        }
        result = _extract_token_from_headers(headers)
        assert result == "api-key"

    def test_falls_back_to_second_candidate_when_first_missing(self):
        """Test that function falls back to second candidate when first is missing."""
        headers = {"x-datarobot-api-token": "Bearer fallback-token"}
        result = _extract_token_from_headers(headers)
        assert result == "fallback-token"

    def test_returns_none_when_no_headers(self):
        """Test that function returns None when no headers are provided."""
        headers = {}
        result = _extract_token_from_headers(headers)
        assert result is None

    def test_returns_none_when_no_matching_headers(self):
        """Test that function returns None when no candidate headers are present."""
        headers = {"content-type": "application/json", "user-agent": "test-client"}
        result = _extract_token_from_headers(headers)
        assert result is None

    def test_handles_empty_token_after_stripping(self):
        """Test that function returns None when token is empty after stripping."""
        headers = {"authorization": "Bearer   "}
        result = _extract_token_from_headers(headers)
        assert result is None

    def test_handles_non_string_header_value(self):
        """Test that function skips non-string header values."""
        headers = {"authorization": 12345}
        result = _extract_token_from_headers(headers)
        assert result is None

    def test_handles_none_header_value(self):
        """Test that function handles None header values."""
        headers = {"authorization": None}
        result = _extract_token_from_headers(headers)
        assert result is None

    def test_strips_whitespace_from_token(self):
        """Test that function strips whitespace from extracted token."""
        headers = {"authorization": "Bearer   token-with-spaces   "}
        result = _extract_token_from_headers(headers)
        assert result == "token-with-spaces"

    def test_handles_bearer_with_multiple_spaces(self):
        """Test that function handles Bearer prefix with multiple spaces."""
        headers = {"authorization": "Bearer  token-123"}
        result = _extract_token_from_headers(headers)
        assert result == "token-123"

    def test_extract_token_from_x_datarobot_authorization_header(self):
        """Test extracting token from x-datarobot-authorization header."""
        headers = {"x-datarobot-authorization": "Bearer user-api-key"}
        result = _extract_token_from_headers(headers)
        assert result == "user-api-key"

    def test_prefers_x_datarobot_authorization_over_authorization(self):
        """Test gateway scenario: x-datarobot-authorization is preferred."""
        headers = {
            "x-datarobot-authorization": "Bearer user-api-key",
            "authorization": "Bearer s2s-jwt-token",
        }
        result = _extract_token_from_headers(headers)
        assert result == "user-api-key"


class TestExtractTokenFromAuthContext:
    """Test cases for _extract_token_from_auth_context function - critical path only."""

    @patch("datarobot_genai.drtools.core.auth.AuthContextHeaderHandler")
    def test_extracts_api_key_from_dr_ctx_metadata(self, mock_handler_class):
        """Test successful extraction of API key from dr_ctx in authorization context metadata."""
        auth_ctx = AuthCtx(
            user=User(id="test-user-123", email="test.user@example.com"),
            identities=[
                Identity(
                    id="identity-1",
                    type="datarobot",
                    provider_type="datarobot_ext_email",
                    provider_user_id="test.user@example.com",
                    provider_identity_id=None,
                ),
            ],
            metadata={"dr_ctx": {"email": "test.user@example.com", "api_key": "test-api-key-789"}},
        )

        mock_handler = Mock()
        mock_handler.get_context.return_value = auth_ctx
        mock_handler_class.return_value = mock_handler

        headers = {"x-datarobot-authorization-context": "jwt-token"}
        result = _extract_token_from_auth_context(headers)

        assert result == "test-api-key-789"
        mock_handler.get_context.assert_called_once_with(headers)

    @patch("datarobot_genai.drtools.core.auth.AuthContextHeaderHandler")
    def test_returns_none_when_no_api_key_in_metadata(self, mock_handler_class):
        """Test that function returns None when auth context has no metadata or no api_key."""
        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler

        # Test with no auth context
        mock_handler.get_context.return_value = None
        assert _extract_token_from_auth_context({}) is None

        # Test with no metadata
        mock_handler.get_context.return_value = AuthCtx(
            user=User(id="test-user-123", email="test.user@example.com"),
            identities=[],
            metadata=None,
        )
        assert _extract_token_from_auth_context({}) is None

        # Test with empty api_key
        mock_handler.get_context.return_value = AuthCtx(
            user=User(id="test-user-123", email="test.user@example.com"),
            identities=[],
            metadata={"dr_ctx": {"email": "test.user@example.com", "api_key": ""}},
        )
        assert _extract_token_from_auth_context({}) is None

    @patch("datarobot_genai.drtools.core.auth.AuthContextHeaderHandler")
    def test_handles_exceptions_gracefully(self, mock_handler_class):
        """Test that function handles exceptions and returns None."""
        mock_handler = Mock()
        mock_handler.get_context.side_effect = ValueError("Invalid JWT")
        mock_handler_class.return_value = mock_handler

        result = _extract_token_from_auth_context(
            {"x-datarobot-authorization-context": "bad-token"}
        )
        assert result is None


class TestExtractTokenFromHeadersWithFallback:
    """Test cases for _extract_token_from_headers_with_fallback - critical path only."""

    def test_prefers_standard_header_over_auth_context(self):
        """Test that standard headers are preferred over auth context metadata."""
        headers = {"authorization": "Bearer standard-token"}

        with patch(
            "datarobot_genai.drtools.core.auth._extract_token_from_auth_context"
        ) as mock_auth_extract:
            result = _extract_token_from_headers_with_fallback(headers)

            assert result == "standard-token"
            mock_auth_extract.assert_not_called()

    @patch("datarobot_genai.drtools.core.auth.AuthContextHeaderHandler")
    def test_falls_back_to_auth_context_when_no_standard_headers(self, mock_handler_class):
        """Test fallback to auth context metadata when standard headers are missing."""
        auth_ctx = AuthCtx(
            user=User(id="test-user-123", email="test.user@example.com"),
            identities=[],
            metadata={"dr_ctx": {"email": "test.user@example.com", "api_key": "fallback-api-key"}},
        )

        mock_handler = Mock()
        mock_handler.get_context.return_value = auth_ctx
        mock_handler_class.return_value = mock_handler

        headers = {"x-datarobot-authorization-context": "jwt-token"}
        result = _extract_token_from_headers_with_fallback(headers)

        assert result == "fallback-api-key"
