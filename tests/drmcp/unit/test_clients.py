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

from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from datarobot.auth.identity import Identity
from datarobot.auth.session import AuthCtx
from datarobot.auth.users import User

from datarobot_genai.drmcp.core.clients import _extract_token_from_auth_context
from datarobot_genai.drmcp.core.clients import _extract_token_from_headers
from datarobot_genai.drmcp.core.clients import dr
from datarobot_genai.drmcp.core.clients import extract_token_from_headers
from datarobot_genai.drmcp.core.clients import get_api_client
from datarobot_genai.drmcp.core.clients import get_s3_bucket_info
from datarobot_genai.drmcp.core.clients import get_sdk_client
from datarobot_genai.drmcp.core.routes_utils import prefix_mount_path


def test_get_sdk_client_returns_dr() -> None:
    mock_creds = MagicMock()
    mock_creds.datarobot.application_api_token = "token"
    mock_creds.datarobot.endpoint = "url"
    with (
        patch("datarobot_genai.drmcp.core.clients.dr.Client") as mock_client,
        patch("datarobot_genai.drmcp.core.clients.get_credentials", return_value=mock_creds),
    ):
        result = get_sdk_client()
        mock_client.assert_called_once_with(token="token", endpoint="url")
        assert result is dr


def test_get_s3_bucket_info() -> None:
    mock_creds = MagicMock()
    mock_creds.aws_predictions_s3_bucket = "bucket"
    mock_creds.aws_predictions_s3_prefix = "prefix"
    with patch("datarobot_genai.drmcp.core.clients.get_credentials", return_value=mock_creds):
        result = get_s3_bucket_info()
        assert result == {"bucket": "bucket", "prefix": "prefix"}


class TestGetApiClient:
    """Test cases for get_api_client function."""

    @patch("datarobot_genai.drmcp.core.clients.get_sdk_client")
    def test_get_api_client_returns_client(self, mock_get_sdk_client):
        """Test that get_api_client returns the REST client."""
        mock_dr = Mock()
        mock_client = Mock()
        mock_rest_client = Mock()
        mock_client.get_client.return_value = mock_rest_client
        mock_dr.client = mock_client
        mock_get_sdk_client.return_value = mock_dr

        result = get_api_client()

        assert result == mock_rest_client
        mock_get_sdk_client.assert_called_once()
        mock_client.get_client.assert_called_once()


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

    def test_prefers_authorization_over_other_headers(self):
        """Test that authorization header is preferred over other candidate headers."""
        headers = {
            "authorization": "Bearer auth-token",
            "x-datarobot-api-token": "Bearer api-token",
            "x-datarobot-api-key": "Bearer api-key",
        }
        result = _extract_token_from_headers(headers)
        assert result == "auth-token"

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


class TestExtractTokenFromAuthContext:
    """Test cases for _extract_token_from_auth_context function - critical path only."""

    @patch("datarobot_genai.drmcp.core.clients.AuthContextHeaderHandler")
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

    @patch("datarobot_genai.drmcp.core.clients.AuthContextHeaderHandler")
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

    @patch("datarobot_genai.drmcp.core.clients.AuthContextHeaderHandler")
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
    """Test cases for extract_token_from_headers - critical path only."""

    def test_prefers_standard_header_over_auth_context(self):
        """Test that standard headers are preferred over auth context metadata."""
        headers = {"authorization": "Bearer standard-token"}

        with patch(
            "datarobot_genai.drmcp.core.clients._extract_token_from_auth_context"
        ) as mock_auth_extract:
            result = extract_token_from_headers(headers)

            assert result == "standard-token"
            mock_auth_extract.assert_not_called()

    @patch("datarobot_genai.drmcp.core.clients.AuthContextHeaderHandler")
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
        result = extract_token_from_headers(headers)

        assert result == "fallback-api-key"


class TestGetSdkClientWithHeaders:
    """Test cases for get_sdk_client function with header extraction."""

    @patch("datarobot_genai.drmcp.core.clients.dr.Client")
    @patch("datarobot_genai.drmcp.core.clients.get_http_headers")
    @patch("datarobot_genai.drmcp.core.clients.get_credentials")
    def test_uses_token_from_authorization_header(
        self, mock_get_creds, mock_get_headers, mock_client
    ):
        """Test that get_sdk_client uses token from Authorization header."""
        mock_get_headers.return_value = {"authorization": "Bearer header-token"}
        mock_creds = MagicMock()
        mock_creds.datarobot.endpoint = "https://test.datarobot.com/api/v2"
        mock_get_creds.return_value = mock_creds

        result = get_sdk_client()

        mock_client.assert_called_once_with(
            token="header-token", endpoint="https://test.datarobot.com/api/v2"
        )
        assert result is dr

    @patch("datarobot_genai.drmcp.core.clients.dr.Client")
    @patch("datarobot_genai.drmcp.core.clients.get_http_headers")
    @patch("datarobot_genai.drmcp.core.clients.get_credentials")
    def test_falls_back_to_credentials_when_no_headers(
        self, mock_get_creds, mock_get_headers, mock_client
    ):
        """Test that get_sdk_client falls back to credentials when no headers."""
        mock_get_headers.return_value = {}
        mock_creds = MagicMock()
        mock_creds.datarobot.application_api_token = "credential-token"
        mock_creds.datarobot.endpoint = "https://test.datarobot.com/api/v2"
        mock_get_creds.return_value = mock_creds

        result = get_sdk_client()

        mock_client.assert_called_once_with(
            token="credential-token", endpoint="https://test.datarobot.com/api/v2"
        )
        assert result is dr

    @patch("datarobot_genai.drmcp.core.clients.dr.Client")
    @patch("datarobot_genai.drmcp.core.clients.get_http_headers")
    @patch("datarobot_genai.drmcp.core.clients.get_credentials")
    def test_falls_back_when_header_extraction_fails(
        self, mock_get_creds, mock_get_headers, mock_client
    ):
        """Test that get_sdk_client falls back when header extraction returns None."""
        mock_get_headers.return_value = {"authorization": "Bearer   "}  # Empty token
        mock_creds = MagicMock()
        mock_creds.datarobot.application_api_token = "credential-token"
        mock_creds.datarobot.endpoint = "https://test.datarobot.com/api/v2"
        mock_get_creds.return_value = mock_creds

        result = get_sdk_client()

        mock_client.assert_called_once_with(
            token="credential-token", endpoint="https://test.datarobot.com/api/v2"
        )
        assert result is dr

    @patch("datarobot_genai.drmcp.core.clients.dr.Client")
    @patch("datarobot_genai.drmcp.core.clients.get_http_headers")
    @patch("datarobot_genai.drmcp.core.clients.get_credentials")
    def test_handles_exception_from_get_http_headers(
        self, mock_get_creds, mock_get_headers, mock_client
    ):
        """Test that get_sdk_client handles exceptions from get_http_headers."""
        mock_get_headers.side_effect = RuntimeError("No HTTP context")
        mock_creds = MagicMock()
        mock_creds.datarobot.application_api_token = "credential-token"
        mock_creds.datarobot.endpoint = "https://test.datarobot.com/api/v2"
        mock_get_creds.return_value = mock_creds

        result = get_sdk_client()

        mock_client.assert_called_once_with(
            token="credential-token", endpoint="https://test.datarobot.com/api/v2"
        )
        assert result is dr

    @patch("datarobot_genai.drmcp.core.clients.dr.Client")
    @patch("datarobot_genai.drmcp.core.clients.get_http_headers")
    @patch("datarobot_genai.drmcp.core.clients.get_credentials")
    def test_uses_x_datarobot_api_token_header(self, mock_get_creds, mock_get_headers, mock_client):
        """Test that get_sdk_client uses x-datarobot-api-token header."""
        mock_get_headers.return_value = {"x-datarobot-api-token": "Bearer api-token-123"}
        mock_creds = MagicMock()
        mock_creds.datarobot.endpoint = "https://test.datarobot.com/api/v2"
        mock_get_creds.return_value = mock_creds

        result = get_sdk_client()

        mock_client.assert_called_once_with(
            token="api-token-123", endpoint="https://test.datarobot.com/api/v2"
        )
        assert result is dr

    @patch("datarobot_genai.drmcp.core.clients.dr.Client")
    @patch("datarobot_genai.drmcp.core.clients.get_http_headers")
    @patch("datarobot_genai.drmcp.core.clients.get_credentials")
    def test_uses_plain_token_without_bearer_prefix(
        self, mock_get_creds, mock_get_headers, mock_client
    ):
        """Test that get_sdk_client handles plain tokens without Bearer prefix."""
        mock_get_headers.return_value = {"authorization": "plain-token-without-bearer"}
        mock_creds = MagicMock()
        mock_creds.datarobot.endpoint = "https://test.datarobot.com/api/v2"
        mock_get_creds.return_value = mock_creds

        result = get_sdk_client()

        mock_client.assert_called_once_with(
            token="plain-token-without-bearer", endpoint="https://test.datarobot.com/api/v2"
        )
        assert result is dr

    @patch("datarobot_genai.drmcp.core.clients.dr.Client")
    @patch("datarobot_genai.drmcp.core.clients.get_http_headers")
    @patch("datarobot_genai.drmcp.core.clients.get_credentials")
    @patch("datarobot_genai.drmcp.core.clients.DRContext")
    def test_resets_dr_context_use_case(
        self, mock_dr_context, mock_get_creds, mock_get_headers, mock_client
    ):
        """Test that get_sdk_client resets DRContext.use_case."""
        mock_get_headers.return_value = {}
        mock_creds = MagicMock()
        mock_creds.datarobot.application_api_token = "token"
        mock_creds.datarobot.endpoint = "https://test.datarobot.com/api/v2"
        mock_get_creds.return_value = mock_creds

        # Set use_case to something
        mock_dr_context.use_case = "some-use-case"

        get_sdk_client()

        # Verify use_case was reset to None
        assert mock_dr_context.use_case is None

    @patch("datarobot_genai.drmcp.core.clients.dr.Client")
    @patch("datarobot_genai.drmcp.core.clients.AuthContextHeaderHandler")
    @patch("datarobot_genai.drmcp.core.clients.get_http_headers")
    @patch("datarobot_genai.drmcp.core.clients.get_credentials")
    def test_extracts_token_from_auth_context_when_no_standard_headers(
        self, mock_get_creds, mock_get_headers, mock_handler_class, mock_client
    ):
        """Test scenario where a user is authenticated via DataRobot
        and the API key is stored in the authorization context metadata rather than in
        standard headers.
        """
        # Create realistic auth context with metadata containing API key
        auth_ctx = AuthCtx(
            user=User(
                id="real-user-456",
                email="jane.smith@company.com",
                name="Jane Smith",
                phone_number=None,
                given_name="Jane",
                family_name="Smith",
                profile_picture_url=None,
                metadata={},
            ),
            identities=[
                Identity(
                    id="dr-identity-1",
                    type="datarobot",
                    provider_type="datarobot_ext_email",
                    provider_user_id="jane.smith@company.com",
                    provider_identity_id=None,
                ),
                Identity(
                    id="google-identity-2",
                    type="oauth2",
                    provider_type="google",
                    provider_user_id="jane.smith@company.com",
                    provider_identity_id="b2c3d4e5-f6a7-8901-bcde-f1234567890a",
                ),
            ],
            metadata={
                "dr_ctx": {
                    "email": "jane.smith@company.com",
                    "api_key": "auth-context-api-key",
                }
            },
        )

        # Mock AuthContextHeaderHandler to return the auth context
        mock_handler = Mock()
        mock_handler.get_context.return_value = auth_ctx
        mock_handler_class.return_value = mock_handler

        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

        # Setup: only auth context header present, no standard auth headers
        mock_get_headers.return_value = {
            "x-datarobot-authorization-context": jwt,
            "content-type": "application/json",
        }

        mock_creds = MagicMock()
        mock_creds.datarobot.endpoint = "https://app.datarobot.com/api/v2"
        mock_get_creds.return_value = mock_creds

        _ = get_sdk_client()

        # Verify the token was extracted from auth context metadata
        mock_client.assert_called_once_with(
            token="auth-context-api-key",
            endpoint="https://app.datarobot.com/api/v2",
        )

        # Verify the auth handler was used to decode the JWT
        mock_handler.get_context.assert_called_once()
        call_args = mock_handler.get_context.call_args[0][0]
        assert call_args["x-datarobot-authorization-context"] == jwt
