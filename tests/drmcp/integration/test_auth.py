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

from typing import Any
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from datarobot.auth.session import AuthCtx

from datarobot_genai.drmcp.core.auth import OAuthMiddleWare
from datarobot_genai.core.utils.auth import AuthContextHeaderHandler


@pytest.fixture
def secret_key() -> str:
    """Return a test secret key for JWT signing."""
    return "test-integration-secret-key"


@pytest.fixture
def auth_context_data() -> dict[str, Any]:
    """Return sample authorization context data matching AuthCtx structure."""
    return {
        "user": {
            "id": "integration_user_123",
            "name": "Integration Test User",
            "email": "integration@test.example.com"
        },
        "identities": [
            {
                "id": "identity_integration_123",
                "type": "user",
                "provider_type": "datarobot",
                "provider_user_id": "integration_user_123"
            }
        ],
        "metadata": {
            "endpoint": "https://app.datarobot.com",
            "account_id": "integration_account_456",
        }
    }


@pytest.fixture
def auth_token(auth_context_data: dict[str, Any], secret_key: str) -> str:
    """Generate a valid JWT token from auth context data."""
    return jwt.encode(auth_context_data, secret_key, algorithm="HS256")


@pytest.fixture
def middleware_with_secret(secret_key: str) -> OAuthMiddleWare:
    """Create an OAuthMiddleware instance for integration testing."""
    with patch("datarobot_genai.drmcp.core.auth.get_config") as mock_config:
        mock_config.return_value.session_secret_key = secret_key
        return OAuthMiddleWare(secret_key=secret_key)


@pytest.mark.asyncio
class TestOAuthMiddlewareIntegration:
    """Integration tests for OAuthMiddleware and AuthContextHeaderHandler."""

    async def test_auth_handler_encode_decode_roundtrip(
        self,
        secret_key: str,
        auth_context_data: dict[str, Any]
    ) -> None:
        """Test that auth handler can encode and decode auth context."""
        handler = AuthContextHeaderHandler(secret_key=secret_key)

        # Encode the auth context into a JWT token
        token = jwt.encode(auth_context_data, secret_key, algorithm="HS256")

        # Decode the token back to auth context
        decoded = handler.decode(token)

        assert decoded is not None
        assert decoded["user"]["id"] == "integration_user_123"
        assert decoded["user"]["name"] == "Integration Test User"
        assert decoded["user"]["email"] == "integration@test.example.com"

    async def test_auth_handler_creates_valid_auth_context(
        self,
        secret_key: str,
        auth_context_data: dict[str, Any],
        auth_token: str
    ) -> None:
        """Test that auth handler creates valid AuthCtx from headers."""
        handler = AuthContextHeaderHandler(secret_key=secret_key)
        headers = {"X-DataRobot-Authorization-Context": auth_token}

        # Extract auth context from headers
        auth_ctx = handler.get_context(headers)

        assert auth_ctx is not None
        assert isinstance(auth_ctx, AuthCtx)
        assert auth_ctx.user.id == "integration_user_123"
        assert auth_ctx.user.name == "Integration Test User"
        assert auth_ctx.user.email == "integration@test.example.com"
        assert len(auth_ctx.identities) == 1
        assert auth_ctx.identities[0].id == "identity_integration_123"

    async def test_auth_handler_handles_missing_header(
        self,
        secret_key: str
    ) -> None:
        """Test that auth handler returns None for missing headers."""
        handler = AuthContextHeaderHandler(secret_key=secret_key)
        headers = {}

        auth_ctx = handler.get_context(headers)

        assert auth_ctx is None

    async def test_auth_handler_handles_invalid_token(
        self,
        secret_key: str
    ) -> None:
        """Test that auth handler returns None for invalid tokens."""
        handler = AuthContextHeaderHandler(secret_key=secret_key)
        headers = {"X-DataRobot-Authorization-Context": "invalid.jwt.token"}

        auth_ctx = handler.get_context(headers)

        assert auth_ctx is None

    async def test_auth_handler_rejects_expired_token(
        self,
        secret_key: str,
        auth_context_data: dict[str, Any]
    ) -> None:
        """Test that auth handler rejects expired tokens."""
        import time

        handler = AuthContextHeaderHandler(secret_key=secret_key)

        # Create an expired token
        expired_payload = {
            **auth_context_data,
            "exp": int(time.time()) - 3600,  # Expired 1 hour ago
            "iat": int(time.time()) - 7200,  # Issued 2 hours ago
        }
        expired_token = jwt.encode(expired_payload, secret_key, algorithm="HS256")
        headers = {"X-DataRobot-Authorization-Context": expired_token}

        auth_ctx = handler.get_context(headers)

        assert auth_ctx is None

    async def test_auth_handler_validates_signature(
        self,
        secret_key: str,
        auth_context_data: dict[str, Any]
    ) -> None:
        """Test that auth handler validates JWT signatures."""
        handler = AuthContextHeaderHandler(secret_key=secret_key)

        # Create token with wrong secret key
        wrong_key = "wrong-secret-key"
        token = jwt.encode(auth_context_data, wrong_key, algorithm="HS256")
        headers = {"X-DataRobot-Authorization-Context": token}

        auth_ctx = handler.get_context(headers)

        assert auth_ctx is None

    async def test_auth_handler_bypass_signature_validation(
        self,
        secret_key: str,
        auth_context_data: dict[str, Any]
    ) -> None:
        """Test that auth handler can bypass signature validation when configured."""
        handler = AuthContextHeaderHandler(
            secret_key=secret_key,
            validate_signature=False
        )

        # Create token with wrong secret key
        wrong_key = "wrong-secret-key"
        token = jwt.encode(auth_context_data, wrong_key, algorithm="HS256")
        headers = {"X-DataRobot-Authorization-Context": token}

        # Should work because signature validation is disabled
        auth_ctx = handler.get_context(headers)

        assert auth_ctx is not None
        assert auth_ctx.user.id == "integration_user_123"

    async def test_middleware_initialization(
        self,
        secret_key: str
    ) -> None:
        """Test that middleware initializes correctly."""
        with patch("datarobot_genai.drmcp.core.auth.get_config") as mock_config:
            mock_config.return_value.session_secret_key = secret_key

            middleware = OAuthMiddleWare(secret_key=secret_key)

            assert middleware.auth_handler is not None
            assert middleware.auth_handler.secret_key == secret_key
            assert middleware.auth_handler.algorithm == "HS256"

    async def test_middleware_extract_auth_context_success(
        self,
        middleware_with_secret: OAuthMiddleWare,
        auth_token: str
    ) -> None:
        """Test that middleware successfully extracts auth context."""
        headers = {"X-DataRobot-Authorization-Context": auth_token}

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            auth_ctx = middleware_with_secret._extract_auth_context()

            assert auth_ctx is not None
            assert isinstance(auth_ctx, AuthCtx)
            assert auth_ctx.user.id == "integration_user_123"

    async def test_middleware_extract_auth_context_missing_header(
        self,
        middleware_with_secret: OAuthMiddleWare
    ) -> None:
        """Test that middleware handles missing auth header."""
        headers = {}

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            auth_ctx = middleware_with_secret._extract_auth_context()

            assert auth_ctx is None

    async def test_middleware_extract_auth_context_invalid_token(
        self,
        middleware_with_secret: OAuthMiddleWare
    ) -> None:
        """Test that middleware handles invalid tokens."""
        headers = {"X-DataRobot-Authorization-Context": "invalid.jwt.token"}

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            auth_ctx = middleware_with_secret._extract_auth_context()

            assert auth_ctx is None

    async def test_middleware_on_call_tool_with_auth_context(
        self,
        middleware_with_secret: OAuthMiddleWare,
        auth_token: str
    ) -> None:
        """Test that middleware attaches auth context to fastmcp_context."""
        from unittest.mock import MagicMock
        from fastmcp.server.middleware import MiddlewareContext
        from fastmcp.tools.tool import ToolResult

        headers = {"X-DataRobot-Authorization-Context": auth_token}

        # Create a mock message and fastmcp_context
        mock_message = MagicMock()
        mock_fastmcp_context = MagicMock()
        context = MiddlewareContext(message=mock_message, fastmcp_context=mock_fastmcp_context)

        # Mock call_next
        mock_result = ToolResult(content="test")
        call_next = AsyncMock(return_value=mock_result)

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware_with_secret.on_call_tool(context, call_next)

            # Verify auth context was attached to fastmcp_context
            assert hasattr(mock_fastmcp_context, "auth_context")
            assert mock_fastmcp_context.auth_context is not None
            assert mock_fastmcp_context.auth_context.user.id == "integration_user_123"

            # Verify call_next was called
            call_next.assert_awaited_once_with(context)

            # Verify result was returned
            assert result is mock_result

    async def test_multiple_auth_contexts_isolation(
        self,
        secret_key: str
    ) -> None:
        """Test that different auth contexts are properly isolated."""
        handler = AuthContextHeaderHandler(secret_key=secret_key)

        # Create two different auth contexts
        auth_data_1 = {
            "user": {"id": "user1", "name": "User One", "email": "user1@example.com"},
            "identities": [{
                "id": "id1",
                "type": "user",
                "provider_type": "datarobot",
                "provider_user_id": "user1"
            }],
        }

        auth_data_2 = {
            "user": {"id": "user2", "name": "User Two", "email": "user2@example.com"},
            "identities": [{
                "id": "id2",
                "type": "user",
                "provider_type": "datarobot",
                "provider_user_id": "user2"
            }],
        }

        token1 = jwt.encode(auth_data_1, secret_key, algorithm="HS256")
        token2 = jwt.encode(auth_data_2, secret_key, algorithm="HS256")

        # Extract first context
        headers1 = {"X-DataRobot-Authorization-Context": token1}
        ctx1 = handler.get_context(headers1)

        # Extract second context
        headers2 = {"X-DataRobot-Authorization-Context": token2}
        ctx2 = handler.get_context(headers2)

        # Verify contexts are different and isolated
        assert ctx1.user.id == "user1"
        assert ctx2.user.id == "user2"
        assert ctx1.user.name == "User One"
        assert ctx2.user.name == "User Two"

