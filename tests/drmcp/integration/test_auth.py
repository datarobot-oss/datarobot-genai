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
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastmcp.server.middleware import MiddlewareContext
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.auth import OAuthMiddleWare


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
                "id": "1234567890",
                "type": "user",
                "provider_type": "github",
                "provider_user_id": "integration_user_123"
            }
        ],
        "metadata": {
            "session_id": "integration_session_789"
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
    """Integration tests for OAuthMiddleware with MiddlewareContext.

    These tests verify the full middleware flow including:
    - Integration with fastmcp MiddlewareContext
    - End-to-end JWT token processing
    - Context propagation through the middleware chain
    """

    async def test_middleware_full_flow_with_valid_auth(
        self,
        middleware_with_secret: OAuthMiddleWare,
        auth_token: str
    ) -> None:
        """Test complete middleware flow: headers -> JWT decode -> AuthCtx -> fastmcp_context.

        This integration test verifies that:
        1. Middleware extracts headers
        2. AuthContextHeaderHandler decodes the JWT token
        3. AuthCtx is properly created from decoded token
        4. AuthCtx is attached to fastmcp_context
        5. Middleware propagates the result correctly
        """

        headers = {"X-DataRobot-Authorization-Context": auth_token}

        # Create real MiddlewareContext (integration test)
        mock_message = MagicMock()
        mock_fastmcp_context = MagicMock()
        context = MiddlewareContext(message=mock_message, fastmcp_context=mock_fastmcp_context)

        # Mock call_next
        expected_result = ToolResult(content="integration_test_result")
        call_next = AsyncMock(return_value=expected_result)

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware_with_secret.on_call_tool(context, call_next)

            # Verify the full integration
            assert mock_fastmcp_context.auth_context is not None, "Auth context should be attached"
            assert mock_fastmcp_context.auth_context.user.id == "integration_user_123"
            assert mock_fastmcp_context.auth_context.user.name == "Integration Test User"
            assert mock_fastmcp_context.auth_context.identities[0].id == "1234567890"

            # Verify middleware didn't break the chain
            call_next.assert_awaited_once_with(context)
            assert result is expected_result

    async def test_middleware_flow_with_missing_auth(
        self,
        middleware_with_secret: OAuthMiddleWare
    ) -> None:
        """Test that middleware gracefully handles missing authentication throughout the full flow."""

        headers = {}  # No auth header

        mock_message = MagicMock()
        mock_fastmcp_context = MagicMock()
        context = MiddlewareContext(message=mock_message, fastmcp_context=mock_fastmcp_context)

        expected_result = ToolResult(content="result_without_auth")
        call_next = AsyncMock(return_value=expected_result)

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware_with_secret.on_call_tool(context, call_next)

            # Verify middleware set auth_context to None
            assert mock_fastmcp_context.auth_context is None

            # Verify middleware still called next handler
            call_next.assert_awaited_once_with(context)
            assert result is expected_result

    async def test_middleware_flow_with_malformed_token(
        self,
        middleware_with_secret: OAuthMiddleWare
    ) -> None:
        """Test that middleware handles malformed tokens gracefully in the full flow."""

        headers = {"X-DataRobot-Authorization-Context": "this.is.not.a.valid.jwt.token"}

        mock_message = MagicMock()
        mock_fastmcp_context = MagicMock()
        context = MiddlewareContext(message=mock_message, fastmcp_context=mock_fastmcp_context)

        expected_result = ToolResult(content="result_with_bad_token")
        call_next = AsyncMock(return_value=expected_result)

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware_with_secret.on_call_tool(context, call_next)

            # Verify middleware handled the error gracefully
            assert mock_fastmcp_context.auth_context is None
            call_next.assert_awaited_once_with(context)
            assert result is expected_result

