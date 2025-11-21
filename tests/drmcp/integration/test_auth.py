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
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

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
            "email": "integration@test.example.com",
        },
        "identities": [
            {
                "id": "1234567890",
                "type": "user",
                "provider_type": "github",
                "provider_user_id": "integration_user_123",
            }
        ],
        "metadata": {"session_id": "integration_session_789"},
    }


@pytest.fixture
def auth_token(auth_context_data: dict[str, Any], secret_key: str) -> str:
    """Generate a valid JWT token from auth context data."""
    return jwt.encode(auth_context_data, secret_key, algorithm="HS256")


@pytest.fixture
def middleware_with_secret(secret_key: str) -> OAuthMiddleWare:
    """Create an OAuthMiddleware instance for integration testing."""
    with patch("datarobot_genai.core.utils.auth.AuthContextConfig") as mock_config_class:
        mock_config = MagicMock()
        mock_config.session_secret_key = secret_key
        mock_config_class.return_value = mock_config
        return OAuthMiddleWare()


@pytest.fixture
def mock_middleware_context() -> MiddlewareContext:
    """Create a mocked MiddlewareContext for testing."""
    mock_message = MagicMock()
    mock_fastmcp_context = MagicMock()
    return MiddlewareContext(message=mock_message, fastmcp_context=mock_fastmcp_context)


@pytest.fixture
def mock_call_next() -> AsyncMock:
    """Create a mocked call_next function that returns a test result."""
    expected_result = ToolResult(content="test_result")
    return AsyncMock(return_value=expected_result)


@pytest.mark.asyncio
class TestOAuthMiddlewareIntegration:
    """Integration tests for OAuthMiddleware with MiddlewareContext.

    These tests verify the full middleware flow including:
    - Integration with fastmcp MiddlewareContext
    - End-to-end JWT token processing
    - Context propagation through the middleware chain
    """

    @pytest.mark.parametrize(
        "header_name",
        ["X-DataRobot-Authorization-Context", "x-datarobot-authorization-context"],
    )
    async def test_middleware_full_flow_with_valid_auth(
        self,
        middleware_with_secret: OAuthMiddleWare,
        auth_token: str,
        header_name: str,
        mock_middleware_context: MiddlewareContext,
        mock_call_next: AsyncMock,
    ) -> None:
        """Test complete middleware flow: headers -> JWT decode -> AuthCtx -> fastmcp_context.

        This integration test verifies that:
        1. Middleware extracts headers
        2. AuthContextHeaderHandler decodes the JWT token
        3. AuthCtx is properly created from decoded token
        4. AuthCtx is stored in fastmcp_context state
        5. Middleware propagates the result correctly
        """
        headers = {header_name: auth_token}

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware_with_secret.on_call_tool(
                mock_middleware_context, mock_call_next
            )

            # Verify auth context was stored with correct key
            mock_fastmcp_context = mock_middleware_context.fastmcp_context
            mock_fastmcp_context.set_state.assert_called_once()
            key, auth_ctx = mock_fastmcp_context.set_state.call_args[0]

            assert key == "authorization_context", "Should use correct state key"
            assert auth_ctx is not None, "Auth context should be attached"
            assert auth_ctx.user.id == "integration_user_123"
            assert auth_ctx.user.name == "Integration Test User"
            assert auth_ctx.identities[0].id == "1234567890"

            # Verify middleware chain continues correctly
            mock_call_next.assert_awaited_once_with(mock_middleware_context)
            assert result == mock_call_next.return_value

    async def test_middleware_flow_with_missing_auth(
        self,
        middleware_with_secret: OAuthMiddleWare,
        mock_middleware_context: MiddlewareContext,
        mock_call_next: AsyncMock,
    ) -> None:
        """Test that middleware gracefully handles missing auth throughout the full flow."""
        headers = {}  # No auth header

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware_with_secret.on_call_tool(
                mock_middleware_context, mock_call_next
            )

            # Verify auth context set to None when missing
            mock_fastmcp_context = mock_middleware_context.fastmcp_context
            mock_fastmcp_context.set_state.assert_called_once()
            key, auth_ctx = mock_fastmcp_context.set_state.call_args[0]

            assert key == "authorization_context"
            assert auth_ctx is None, "Auth context should be None when missing"

            # Verify middleware chain continues
            mock_call_next.assert_awaited_once_with(mock_middleware_context)
            assert result == mock_call_next.return_value

    async def test_middleware_flow_with_malformed_token(
        self,
        middleware_with_secret: OAuthMiddleWare,
        mock_middleware_context: MiddlewareContext,
        mock_call_next: AsyncMock,
    ) -> None:
        """Test that middleware handles malformed tokens gracefully in the full flow."""
        headers = {"X-DataRobot-Authorization-Context": "this.is.not.a.valid.jwt.token"}

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware_with_secret.on_call_tool(
                mock_middleware_context, mock_call_next
            )

            # Verify auth context set to None for malformed token
            mock_fastmcp_context = mock_middleware_context.fastmcp_context
            mock_fastmcp_context.set_state.assert_called_once()
            key, auth_ctx = mock_fastmcp_context.set_state.call_args[0]

            assert key == "authorization_context"
            assert auth_ctx is None, "Auth context should be None for malformed token"

            # Verify middleware chain continues
            mock_call_next.assert_awaited_once_with(mock_middleware_context)
            assert result == mock_call_next.return_value
