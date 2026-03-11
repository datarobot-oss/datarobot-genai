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
from fastmcp.server.context import Context
from fastmcp.server.middleware import MiddlewareContext
from fastmcp.tools.tool import ToolResult

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler
from datarobot_genai.drmcp.core.auth import OAuthMiddleWare
from datarobot_genai.drmcp.core.auth import must_get_auth_context


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
    middleware_instance = OAuthMiddleWare(auth_handler=auth_handler)
    return middleware_instance


@pytest.fixture
def middleware_context() -> MiddlewareContext:
    """Create a mock middleware context with fastmcp_context."""
    context = MagicMock(spec=MiddlewareContext)

    # Create a dict to store state
    state_storage = {}

    def set_state(key: str, value: Any) -> None:
        state_storage[key] = value

    def get_state(key: str) -> Any:
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
            auth_context = middleware_context.fastmcp_context.get_state("authorization_context")
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
            assert middleware_context.fastmcp_context.get_state("authorization_context") is None

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
            assert middleware_context.fastmcp_context.get_state("authorization_context") is None

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
            assert middleware_context.fastmcp_context.get_state("authorization_context") is not None
            assert isinstance(
                middleware_context.fastmcp_context.get_state("authorization_context"), AuthCtx
            )

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
            assert middleware_context.fastmcp_context.get_state("authorization_context") is None

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

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, mock_next)

            # Verify the exact result is returned
            assert result is expected_result
            assert result.structured_content == {"key": "value", "result": "Custom result"}

    async def test_middleware_initialization(self, secret_key: str) -> None:
        """Test that middleware initializes correctly with config."""
        middleware = OAuthMiddleWare(AuthContextHeaderHandler(secret_key))

        assert middleware.auth_handler is not None
        assert middleware.auth_handler.secret_key == secret_key
        assert middleware.auth_handler.algorithm == "HS256"

    async def test_middleware_initialization_without_secret_key(self) -> None:
        """Test that middleware initializes correctly without explicit secret key."""
        with patch("datarobot_genai.core.utils.auth.AuthContextConfig") as mock_config_class:
            mock_config_instance = MagicMock()
            mock_config_instance.session_secret_key = "config-secret-key"
            mock_config_class.return_value = mock_config_instance

            middleware = OAuthMiddleWare()

            assert middleware.auth_handler is not None
            assert middleware.auth_handler.secret_key == "config-secret-key"

    async def test_middleware_with_lowercase_header_name(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
        auth_token: str,
    ) -> None:
        """Test that middleware handles lowercase header names (as returned by get_http_headers)."""
        headers = {"x-datarobot-authorization-context": auth_token}

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, call_next)

            call_next.assert_awaited_once_with(middleware_context)
            assert isinstance(result, ToolResult)

            # Verify auth_context was attached despite lowercase header name
            auth_context = middleware_context.fastmcp_context.get_state("authorization_context")
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

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
            result = await middleware.on_call_tool(middleware_context, call_next)

            call_next.assert_awaited_once_with(middleware_context)
            assert isinstance(result, ToolResult)

            # Verify auth_context was attached
            auth_context = middleware_context.fastmcp_context.get_state("authorization_context")
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

        with patch("datarobot_genai.drmcp.core.auth.get_http_headers", return_value=headers):
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
        with patch("datarobot_genai.drmcp.core.auth.get_http_headers") as mock_get_headers:
            mock_get_headers.side_effect = ValueError("Invalid header format")

            result = await middleware.on_call_tool(middleware_context, call_next)

            call_next.assert_awaited_once_with(middleware_context)
            assert isinstance(result, ToolResult)

            # Verify auth_context is None due to exception
            assert middleware_context.fastmcp_context.get_state("authorization_context") is None

    async def test_middleware_handles_key_error_exception(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
    ) -> None:
        """Test that middleware handles KeyError exceptions gracefully."""
        with patch("datarobot_genai.drmcp.core.auth.get_http_headers") as mock_get_headers:
            mock_get_headers.side_effect = KeyError("Missing key")

            result = await middleware.on_call_tool(middleware_context, call_next)

            call_next.assert_awaited_once_with(middleware_context)
            assert isinstance(result, ToolResult)

            # Verify auth_context is None due to exception
            assert middleware_context.fastmcp_context.get_state("authorization_context") is None

    async def test_middleware_handles_type_error_exception(
        self,
        middleware: OAuthMiddleWare,
        middleware_context: MiddlewareContext,
        call_next: AsyncMock,
    ) -> None:
        """Test that middleware handles TypeError exceptions gracefully."""
        with patch("datarobot_genai.drmcp.core.auth.get_http_headers") as mock_get_headers:
            mock_get_headers.side_effect = TypeError("Invalid type")

            result = await middleware.on_call_tool(middleware_context, call_next)

            call_next.assert_awaited_once_with(middleware_context)
            assert isinstance(result, ToolResult)

            # Verify auth_context is None due to exception
            assert middleware_context.fastmcp_context.get_state("authorization_context") is None


class TestGetAuthContext:
    """Test cases for must_get_auth_context function."""

    @pytest.mark.asyncio
    async def test_get_auth_context_success(self) -> None:
        """Test that must_get_auth_context returns auth context when available.

        This test mocks at the fastmcp.server.context level to simulate a real
        Context with state management.
        """
        mock_fastmcp = MagicMock()
        context = Context(mock_fastmcp)

        # Create a real AuthCtx object to store in state
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

        # Set the auth context in the context state
        context.set_state("authorization_context", auth_ctx)

        # Patch get_context to return our context with the auth context in state
        with patch("datarobot_genai.drmcp.core.auth.get_context", return_value=context):
            result = await must_get_auth_context()

            # Verify the auth context was retrieved correctly
            assert result is auth_ctx
            assert result.user.id == "user123"
            assert result.user.name == "Test User"
            assert result.user.email == "test@example.com"
            assert len(result.identities) == 1
            assert result.identities[0].provider_type == "datarobot"

    @pytest.mark.asyncio
    async def test_get_auth_context_raises_when_missing(self) -> None:
        """Test that must_get_auth_context raises RuntimeError when auth context is missing.

        This test uses a real Context object without any auth context set in state.
        """
        mock_fastmcp = MagicMock()
        context = Context(mock_fastmcp)

        # Don't set any auth context - state will be empty
        with patch("datarobot_genai.drmcp.core.auth.get_context", return_value=context):
            with pytest.raises(
                RuntimeError,
                match="Could not retrieve authorization context from FastMCP context state.",
            ):
                await must_get_auth_context()

    @pytest.mark.asyncio
    async def test_get_auth_context_raises_when_no_context(self) -> None:
        """Test that RuntimeError is raised when no active context exists.

        This simulates the case where get_context() is called outside of a
        FastMCP request context.
        """
        with patch("datarobot_genai.drmcp.core.auth.get_context") as mock_get_context:
            # Simulate no active context
            mock_get_context.side_effect = RuntimeError("No active context found.")

            with pytest.raises(RuntimeError, match="No active context found."):
                await must_get_auth_context()

    @pytest.mark.asyncio
    async def test_get_auth_context_with_state_isolation(self) -> None:
        """Test that auth context is properly isolated in context state.

        This test verifies that different contexts maintain separate state,
        which is important for concurrent request handling.
        """
        mock_fastmcp = MagicMock()
        context1 = Context(mock_fastmcp)
        context2 = Context(mock_fastmcp)

        # Create different auth contexts with different users
        auth_ctx1 = AuthCtx(
            user=User(id="user1", name="Alice", email="alice@example.com"),
            identities=[
                Identity(
                    id="identity1",
                    type="user",
                    provider_type="datarobot",
                    provider_user_id="user1",
                )
            ],
            metadata={"endpoint": "https://app.datarobot.com"},
        )

        auth_ctx2 = AuthCtx(
            user=User(id="user2", name="Bob", email="bob@example.com"),
            identities=[
                Identity(
                    id="identity2",
                    type="user",
                    provider_type="datarobot",
                    provider_user_id="user2",
                )
            ],
            metadata={"endpoint": "https://app.datarobot.com"},
        )

        # Set different auth contexts in each context
        context1.set_state("authorization_context", auth_ctx1)
        context2.set_state("authorization_context", auth_ctx2)

        # Verify context1 returns the correct auth context
        with patch("datarobot_genai.drmcp.core.auth.get_context", return_value=context1):
            result1 = await must_get_auth_context()
            assert result1.user.id == "user1"
            assert result1.user.name == "Alice"

        # Verify context2 returns its own auth context (state isolation)
        with patch("datarobot_genai.drmcp.core.auth.get_context", return_value=context2):
            result2 = await must_get_auth_context()
            assert result2.user.id == "user2"
            assert result2.user.name == "Bob"

    @pytest.mark.asyncio
    async def test_get_auth_context_retrieves_correct_key(self) -> None:
        """Test that must_get_auth_context retrieves the correct state key.

        This ensures we're using the right key ('authorization_context') and not
        accidentally retrieving other state values.
        """
        mock_fastmcp = MagicMock()
        context = Context(mock_fastmcp)

        # Set multiple state values to ensure we retrieve the correct one
        context.set_state("other_key", "other_value")
        context.set_state("another_key", {"some": "data"})

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
            metadata={},
        )
        context.set_state("authorization_context", auth_ctx)

        with patch("datarobot_genai.drmcp.core.auth.get_context", return_value=context):
            result = await must_get_auth_context()

            # Verify we got the auth context, not other state values
            assert result is auth_ctx
            assert isinstance(result, AuthCtx)
            assert result.user.id == "user123"
            # Ensure we didn't get other state values
            assert result != "other_value"
            assert result != {"some": "data"}

    @pytest.mark.asyncio
    async def test_get_auth_context_with_complex_metadata(self) -> None:
        """Test that auth context with complex metadata is properly preserved.

        This verifies that all fields of AuthCtx are properly stored and retrieved
        from the context state.
        """
        mock_fastmcp = MagicMock()
        context = Context(mock_fastmcp)

        # Create an auth context with complex metadata
        auth_ctx = AuthCtx(
            user=User(id="user456", name="Complex User", email="complex@example.com"),
            identities=[
                Identity(
                    id="identity1",
                    type="oauth2",
                    provider_type="provider1",
                    provider_user_id="oauth-user-123",
                ),
                Identity(
                    id="identity2",
                    type="datarobot",
                    provider_type="provider2",
                    provider_user_id="service-account-456",
                ),
            ],
            metadata={
                "endpoint": "https://app.datarobot.com",
                "account_id": "account789",
                "permissions": ["read", "write", "execute"],
                "custom_data": {"key1": "value1", "key2": 42},
            },
        )

        context.set_state("authorization_context", auth_ctx)

        with patch("datarobot_genai.drmcp.core.auth.get_context", return_value=context):
            result = await must_get_auth_context()

            # Verify all fields are preserved
            assert result is auth_ctx
            assert result.user.id == "user456"
            assert len(result.identities) == 2
            assert result.identities[0].provider_type == "provider1"
            assert result.identities[1].provider_type == "provider2"
            assert result.metadata["endpoint"] == "https://app.datarobot.com"
            assert result.metadata["account_id"] == "account789"
            assert "permissions" in result.metadata
            assert result.metadata["custom_data"]["key2"] == 42
