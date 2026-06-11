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
from datarobot.auth.identity import Identity
from datarobot.auth.session import AuthCtx
from datarobot.auth.users import User
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import MiddlewareContext
from fastmcp.tools.tool import ToolResult

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler
from datarobot_genai.drmcp.core.middleware import create_oauth_middleware
from datarobot_genai.drmcpbase.middleware import OAuthMiddleWare
from datarobot_genai.drmcpbase.middleware import read_http_headers
from datarobot_genai.drmcputils.auth import must_get_auth_context
from datarobot_genai.drmcputils.auth import set_auth_context
from datarobot_genai.drmcputils.auth import set_request_headers


@pytest.fixture
def secret_key() -> str:
    """Return a test secret key for JWT signing."""
    return "test-secret-key"


@pytest.fixture
def auth_context_data() -> dict[str, Any]:
    """Return sample authorization context data."""
    return {
        "user": {"id": "user123", "name": "Test User", "email": "test@example.com"},
        "identities": [
            {
                "id": "identity123",
                "type": "user",
                "provider_type": "datarobot",
                "provider_user_id": "user123",
            }
        ],
        "metadata": {
            "endpoint": "https://app.datarobot.com",
            "account_id": "account456",
        },
    }


@pytest.fixture
def auth_token(auth_context_data: dict[str, Any], secret_key: str) -> str:
    """Generate a valid JWT token from auth context data."""
    return jwt.encode(auth_context_data, secret_key, algorithm="HS256")


@pytest.fixture
def middleware(secret_key: str) -> OAuthMiddleWare:
    """Create an OAuthMiddleware instance with mocked config."""
    auth_handler = AuthContextHeaderHandler(secret_key=secret_key, algorithm="HS256")
    middleware_instance = create_oauth_middleware(
        extract_auth_context=auth_handler.get_context,
    )
    return middleware_instance


@pytest.fixture
def middleware_context() -> MiddlewareContext:
    """Create a mock middleware context with fastmcp_context."""
    context = MagicMock(spec=MiddlewareContext)

    # Create a dict to store state (set_state/get_state are async in fastmcp 3.x)
    state_storage: dict[str, Any] = {}

    async def set_state(key: str, value: Any) -> None:
        state_storage[key] = value

    async def get_state(key: str) -> Any:
        return state_storage.get(key)

    context.fastmcp_context = MagicMock()
    context.fastmcp_context.set_state = set_state
    context.fastmcp_context.get_state = get_state

    return context


@pytest.fixture
def call_next() -> AsyncMock:
    """Create a mock call_next function that returns a successful tool result."""
    mock_next = AsyncMock()
    mock_next.return_value = ToolResult(structured_content={"result": "Success"})
    return mock_next


class TestReadHttpHeaders:
    """Test that middleware reads authorization headers via include_all=True."""

    def test_fastmcp_strips_auth_headers_by_default(self) -> None:
        """Confirm fastmcp 3.x strips authorization without include_all."""
        fake_request = MagicMock()
        fake_request.headers = MagicMock()
        fake_request.headers.items.return_value = [
            ("authorization", "Bearer token123"),
            ("x-datarobot-api-token", "dr-token-456"),
            ("x-custom", "keep-me"),
        ]

        with patch("fastmcp.server.dependencies.get_http_request", return_value=fake_request):
            default_headers = get_http_headers()
            all_headers = get_http_headers(include_all=True)

        assert "authorization" not in default_headers
        assert "x-custom" in default_headers
        assert "authorization" in all_headers
        assert all_headers["authorization"] == "Bearer token123"

    def test_middleware_read_http_headers_passes_include_all(self) -> None:
        """Verify middleware _read_http_headers passes include_all=True."""
        with patch("datarobot_genai.drmcpbase.middleware.get_http_headers") as mock_get_headers:
            mock_get_headers.return_value = {"authorization": "Bearer token123"}
            result = read_http_headers()

            mock_get_headers.assert_called_once_with(include_all=True)
            assert result == {"authorization": "Bearer token123"}


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

        with patch("datarobot_genai.drmcpbase.middleware.read_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, call_next)

            # Verify call_next was called
            call_next.assert_awaited_once_with(middleware_context)

            # Verify result is returned
            assert isinstance(result, ToolResult)

            # Verify auth_context was attached to fastmcp_context
            auth_context = await middleware_context.fastmcp_context.get_state(
                "authorization_context"
            )
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

        with patch("datarobot_genai.drmcpbase.middleware.read_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, call_next)

            # Verify call_next was still called
            call_next.assert_awaited_once_with(middleware_context)

            # Verify result is returned
            assert isinstance(result, ToolResult)

            # Verify auth_context is None
            assert (
                await middleware_context.fastmcp_context.get_state("authorization_context") is None
            )

    async def test_on_call_tool_with_invalid_auth_header(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
    ) -> None:
        """Test that middleware handles invalid auth token gracefully."""
        headers = {"X-DataRobot-Authorization-Context": "invalid.jwt.token"}

        with patch("datarobot_genai.drmcpbase.middleware.read_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, call_next)

            # Verify call_next was still called (middleware doesn't block execution)
            call_next.assert_awaited_once_with(middleware_context)

            # Verify result is returned
            assert isinstance(result, ToolResult)

            # Verify auth_context is None due to invalid token
            assert (
                await middleware_context.fastmcp_context.get_state("authorization_context") is None
            )

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

        with patch("datarobot_genai.drmcpbase.middleware.read_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, call_next)

            # Verify successful processing
            call_next.assert_awaited_once_with(middleware_context)
            assert isinstance(result, ToolResult)

            # Verify auth_context was properly extracted
            assert (
                await middleware_context.fastmcp_context.get_state("authorization_context")
                is not None
            )
            assert isinstance(
                await middleware_context.fastmcp_context.get_state("authorization_context"), AuthCtx
            )

    async def test_on_call_tool_exception_handling(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
    ) -> None:
        """Test that middleware handles exceptions during header parsing gracefully."""
        with patch("datarobot_genai.drmcpbase.middleware.read_http_headers") as mock_headers:
            # Simulate an exception when getting headers
            mock_headers.side_effect = RuntimeError("Header parsing failed")

            result = await middleware.on_call_tool(middleware_context, call_next)

            # Verify call_next was still called despite the exception
            call_next.assert_awaited_once_with(middleware_context)

            # Verify result is returned
            assert isinstance(result, ToolResult)

            # Verify auth_context is None due to exception
            assert (
                await middleware_context.fastmcp_context.get_state("authorization_context") is None
            )

    async def test_on_call_tool_propagates_tool_result(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        auth_token: str,
    ) -> None:
        """Test that middleware correctly propagates the tool result from call_next."""
        headers = {"X-DataRobot-Authorization-Context": auth_token}
        expected_result = ToolResult(structured_content={"key": "value", "result": "Custom result"})

        mock_next = AsyncMock(return_value=expected_result)

        with patch("datarobot_genai.drmcpbase.middleware.read_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, mock_next)

            # Verify the exact result is returned
            assert result is expected_result
            assert result.structured_content == {"key": "value", "result": "Custom result"}

    async def test_middleware_initialization(self, secret_key: str) -> None:
        """Test that middleware initializes correctly with config."""
        auth_handler = AuthContextHeaderHandler(secret_key=secret_key, algorithm="HS256")
        middleware = create_oauth_middleware(extract_auth_context=auth_handler.get_context)

        assert isinstance(middleware, OAuthMiddleWare)

    async def test_middleware_initialization_without_secret_key(self) -> None:
        """Test that middleware initializes correctly without explicit secret key."""
        with patch("datarobot_genai.core.utils.auth.AuthContextConfig") as mock_config_class:
            mock_config_instance = MagicMock()
            mock_config_instance.session_secret_key = "config-secret-key"
            mock_config_class.return_value = mock_config_instance

            middleware = create_oauth_middleware()

            assert isinstance(middleware, OAuthMiddleWare)

    async def test_middleware_with_lowercase_header_name(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
        auth_token: str,
    ) -> None:
        """Test that middleware handles lowercase header names (as returned by get_http_headers)."""
        headers = {"x-datarobot-authorization-context": auth_token}

        with patch("datarobot_genai.drmcpbase.middleware.read_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, call_next)

            call_next.assert_awaited_once_with(middleware_context)
            assert isinstance(result, ToolResult)

            # Verify auth_context was attached despite lowercase header name
            auth_context = await middleware_context.fastmcp_context.get_state(
                "authorization_context"
            )
            assert auth_context is not None
            assert isinstance(auth_context, AuthCtx)
            assert auth_context.user.id == "user123"

    async def test_middleware_with_mixed_case_header_name(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
        auth_token: str,
    ) -> None:
        """Test that middleware handles mixed case header names."""
        headers = {"X-DataRobot-Authorization-Context": auth_token}

        with patch("datarobot_genai.drmcpbase.middleware.read_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, call_next)

            call_next.assert_awaited_once_with(middleware_context)
            assert isinstance(result, ToolResult)

            # Verify auth_context was attached
            auth_context = await middleware_context.fastmcp_context.get_state(
                "authorization_context"
            )
            assert auth_context is not None
            assert isinstance(auth_context, AuthCtx)

    async def test_middleware_with_no_fastmcp_context(
        self,
        middleware: OAuthMiddleWare,
        call_next: AsyncMock,
        auth_token: str,
    ) -> None:
        """Test that middleware handles missing fastmcp_context gracefully."""
        context = MagicMock(spec=MiddlewareContext)
        context.fastmcp_context = None  # No fastmcp_context

        headers = {"X-DataRobot-Authorization-Context": auth_token}

        with patch("datarobot_genai.drmcpbase.middleware.read_http_headers", return_value=headers):
            result = await middleware.on_call_tool(context, call_next)

            # Should still call next and return result
            call_next.assert_awaited_once_with(context)
            assert isinstance(result, ToolResult)

    async def test_middleware_handles_value_error_exception(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
    ) -> None:
        """Test that middleware handles ValueError exceptions gracefully."""
        with patch("datarobot_genai.drmcpbase.middleware.read_http_headers") as mock_get_headers:
            mock_get_headers.side_effect = ValueError("Invalid header format")

            result = await middleware.on_call_tool(middleware_context, call_next)

            call_next.assert_awaited_once_with(middleware_context)
            assert isinstance(result, ToolResult)

            # Verify auth_context is None due to exception
            assert (
                await middleware_context.fastmcp_context.get_state("authorization_context") is None
            )

    async def test_middleware_handles_key_error_exception(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
    ) -> None:
        """Test that middleware handles KeyError exceptions gracefully."""
        with patch("datarobot_genai.drmcpbase.middleware.read_http_headers") as mock_get_headers:
            mock_get_headers.side_effect = KeyError("Missing key")

            result = await middleware.on_call_tool(middleware_context, call_next)

            call_next.assert_awaited_once_with(middleware_context)
            assert isinstance(result, ToolResult)

            # Verify auth_context is None due to exception
            assert (
                await middleware_context.fastmcp_context.get_state("authorization_context") is None
            )

    async def test_middleware_handles_type_error_exception(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
    ) -> None:
        """Test that middleware handles TypeError exceptions gracefully."""
        with patch("datarobot_genai.drmcpbase.middleware.read_http_headers") as mock_get_headers:
            mock_get_headers.side_effect = TypeError("Invalid type")

            result = await middleware.on_call_tool(middleware_context, call_next)

            call_next.assert_awaited_once_with(middleware_context)
            assert isinstance(result, ToolResult)

            # Verify auth_context is None due to exception
            assert (
                await middleware_context.fastmcp_context.get_state("authorization_context") is None
            )


class TestGetAuthContext:
    """Test cases for must_get_auth_context function."""

    @pytest.fixture(autouse=True)
    def _clear_auth_context(self) -> None:
        set_auth_context(None)
        set_request_headers({})
        yield
        set_auth_context(None)
        set_request_headers({})

    @pytest.mark.asyncio
    async def test_get_auth_context_success(self) -> None:
        auth_ctx = AuthCtx(
            user=User(id="user123", name="Test User", email="test@example.com"),
            identities=[
                Identity(
                    id="identity123",
                    type="user",
                    provider_type="datarobot",
                    provider_user_id="user123",
                )
            ],
            metadata={"endpoint": "https://app.datarobot.com", "account_id": "account456"},
        )
        set_auth_context(auth_ctx)
        result = await must_get_auth_context()
        assert result is auth_ctx
        assert result.user.id == "user123"

    @pytest.mark.asyncio
    async def test_get_auth_context_raises_when_missing(self) -> None:
        with pytest.raises(RuntimeError, match="Could not retrieve authorization context."):
            await must_get_auth_context()

    @pytest.mark.asyncio
    async def test_get_auth_context_reconstructs_from_dict(self) -> None:
        set_auth_context(
            {
                "user": {"id": "user123", "name": "Test User", "email": "test@example.com"},
                "identities": [],
                "metadata": {},
            }
        )
        result = await must_get_auth_context()
        assert result.user.id == "user123"

    @pytest.mark.asyncio
    async def test_get_auth_context_with_complex_metadata(self) -> None:
        auth_ctx = AuthCtx(
            user=User(id="user456", name="Complex User", email="complex@example.com"),
            identities=[
                Identity(
                    id="identity1",
                    type="oauth2",
                    provider_type="provider1",
                    provider_user_id="oauth-user-123",
                ),
            ],
            metadata={
                "endpoint": "https://app.datarobot.com",
                "account_id": "account789",
                "permissions": ["read", "write", "execute"],
                "custom_data": {"key1": "value1", "key2": 42},
            },
        )
        set_auth_context(auth_ctx)
        result = await must_get_auth_context()
        assert result.user.id == "user456"
        assert result.metadata["custom_data"]["key2"] == 42
