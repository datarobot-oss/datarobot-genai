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
from datarobot.auth.session import AuthCtx
from fastmcp.server.middleware import MiddlewareContext
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drmcp.core.auth import OAuthMiddleWare


@pytest.fixture
def secret_key() -> str:
    """Return a test secret key for JWT signing."""
    return "test-secret-key"


@pytest.fixture
def auth_context_data() -> dict[str, Any]:
    """Return sample authorization context data."""
    return {
        "user": {
            "id": "user123",
            "name": "Test User",
            "email": "test@example.com"
        },
        "identities": [
            {
                "id": "identity123",
                "type": "user",
                "provider_type": "datarobot",
                "provider_user_id": "user123"
            }
        ],
        "metadata": {
            "endpoint": "https://app.datarobot.com",
            "account_id": "account456",
        }
    }


@pytest.fixture
def auth_token(auth_context_data: dict[str, Any], secret_key: str) -> str:
    """Generate a valid JWT token from auth context data."""
    return jwt.encode(auth_context_data, secret_key, algorithm="HS256")


@pytest.fixture
def middleware(secret_key: str) -> OAuthMiddleWare:
    """Create an OAuthMiddleware instance with mocked config."""
    return OAuthMiddleWare(secret_key)


@pytest.fixture
def middleware_context() -> MiddlewareContext:
    """Create a mock middleware context with fastmcp_context."""
    context = MagicMock(spec=MiddlewareContext)
    context.fastmcp_context = MagicMock()
    return context


@pytest.fixture
def call_next() -> AsyncMock:
    """Create a mock call_next function that returns a successful tool result."""
    mock_next = AsyncMock()
    mock_next.return_value = ToolResult(content="Success")
    return mock_next


class TestOAuthMiddleware:
    """Tests for OAuthMiddleware class."""

    async def test_on_call_tool_with_valid_auth_header(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
        auth_token: str,
    ) -> None:
        """Test that middleware successfully parses valid auth header and attaches context."""
        headers = {"X-DataRobot-Authorization-Context": auth_token}

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, call_next)

            # Verify call_next was called
            call_next.assert_awaited_once_with(middleware_context)

            # Verify result is returned
            assert isinstance(result, ToolResult)

            # Verify auth_context was attached to fastmcp_context
            auth_context = middleware_context.fastmcp_context.auth_context
            assert auth_context is not None
            assert isinstance(auth_context, AuthCtx)
            assert auth_context.user.id == "user123"
            assert auth_context.user.name == "Test User"

    async def test_on_call_tool_without_auth_header(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
    ) -> None:
        """Test that middleware handles missing auth header gracefully."""
        headers = {}

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, call_next)

            # Verify call_next was still called
            call_next.assert_awaited_once_with(middleware_context)

            # Verify result is returned
            assert isinstance(result, ToolResult)

            # Verify auth_context is None
            assert middleware_context.fastmcp_context.auth_context is None

    async def test_on_call_tool_with_invalid_auth_header(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
    ) -> None:
        """Test that middleware handles invalid auth token gracefully."""
        headers = {"X-DataRobot-Authorization-Context": "invalid.jwt.token"}

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, call_next)

            # Verify call_next was still called (middleware doesn't block execution)
            call_next.assert_awaited_once_with(middleware_context)

            # Verify result is returned
            assert isinstance(result, ToolResult)

            # Verify auth_context is None due to invalid token
            assert middleware_context.fastmcp_context.auth_context is None

    async def test_on_call_tool_with_multiple_headers(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
        auth_token: str,
    ) -> None:
        """Test that middleware extracts auth header among other headers."""
        headers = {
            "Content-Type": "application/json",
            "X-DataRobot-Authorization-Context": auth_token,
            "User-Agent": "test-client",
        }

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, call_next)

            # Verify successful processing
            call_next.assert_awaited_once_with(middleware_context)
            assert isinstance(result, ToolResult)

            # Verify auth_context was properly extracted
            assert middleware_context.fastmcp_context.auth_context is not None
            assert isinstance(middleware_context.fastmcp_context.auth_context, AuthCtx)

    async def test_on_call_tool_exception_handling(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
    ) -> None:
        """Test that middleware handles exceptions during header parsing gracefully."""
        with patch("datarobot_genai.drmcp.core.auth.get_http_headers") as mock_headers:
            # Simulate an exception when getting headers
            mock_headers.side_effect = RuntimeError("Header parsing failed")

            result = await middleware.on_call_tool(middleware_context, call_next)

            # Verify call_next was still called despite the exception
            call_next.assert_awaited_once_with(middleware_context)

            # Verify result is returned
            assert isinstance(result, ToolResult)

            # Verify auth_context is None due to exception
            assert middleware_context.fastmcp_context.auth_context is None

    async def test_on_call_tool_propagates_tool_result(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        auth_token: str,
    ) -> None:
        """Test that middleware correctly propagates the tool result from call_next."""
        headers = {"X-DataRobot-Authorization-Context": auth_token}
        expected_result = ToolResult(
            structured_content={"key": "value", "result": "Custom result"}
        )

        mock_next = AsyncMock(return_value=expected_result)

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, mock_next)

            # Verify the exact result is returned
            assert result is expected_result
            assert result.structured_content == {"key": "value", "result": "Custom result"}

    async def test_middleware_initialization(self, secret_key: str) -> None:
        """Test that middleware initializes correctly with config."""
        middleware = OAuthMiddleWare(secret_key)

        assert middleware.auth_handler is not None
        assert middleware.auth_handler.secret_key == secret_key
        assert middleware.auth_handler.algorithm == "HS256"

