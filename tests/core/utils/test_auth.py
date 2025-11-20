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
import binascii
import os
import random
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

import jwt
import pytest
from datarobot.auth.identity import Identity
from datarobot.auth.session import AuthCtx
from datarobot.auth.users import User
from datarobot.models.genai.agent.auth import ToolAuth
from datarobot.models.genai.agent.auth import set_authorization_context

from datarobot_genai.core.utils.auth import AsyncOAuthTokenProvider
from datarobot_genai.core.utils.auth import AuthContextHeaderHandler


@pytest.fixture
def auth_context() -> dict[str, Any]:
    """Return a sample authorization context that can be serialized/deserialized."""
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
def secret_key() -> str:
    """Return a repeatable sample secret key for signing JWTs without mutating global RNG."""
    rnd = random.Random(42)
    key_bytes = bytes(rnd.getrandbits(8) for _ in range(64))
    return binascii.hexlify(key_bytes).decode("utf-8")


@pytest.fixture
def handler(secret_key: str) -> AuthContextHeaderHandler:
    """Return an AuthContextHeaderHandler instance with a secret key."""
    return AuthContextHeaderHandler(secret_key=secret_key)


@pytest.fixture
def handler_no_verification(secret_key: str) -> AuthContextHeaderHandler:
    """Return an AuthContextHeaderHandler instance without signature verification."""
    return AuthContextHeaderHandler(secret_key=secret_key, validate_signature=False)


class TestAuthContextHeaderHandlerEncode:
    """Tests for encoding authorization context into JWT tokens."""

    def test_reject_none_algorithm(self, secret_key: str) -> None:
        """Test that 'none' algorithm is rejected to prevent algorithm confusion attacks."""
        with pytest.raises(ValueError, match="Algorithm None is not allowed"):
            AuthContextHeaderHandler(secret_key=secret_key, algorithm=None)

    def test_encode_with_valid_context(
        self, handler: AuthContextHeaderHandler, auth_context: dict[str, Any]
    ) -> None:
        """Test encoding a valid authorization context."""
        with patch(
            "datarobot_genai.core.utils.auth.get_authorization_context", return_value=auth_context
        ):
            token = handler.encode()

            assert isinstance(token, str), "Token should be a string"
            assert token, "Token should not be empty"

    def test_encode_with_no_context(self, handler: AuthContextHeaderHandler) -> None:
        """Test encoding when no authorization context is available."""
        with patch("datarobot_genai.core.utils.auth.get_authorization_context", return_value=None):
            token = handler.encode()

            assert token is None, "Token should be None when no context is available"

    def test_encode_with_no_secret_key_raises_warning(self, auth_context: dict[str, Any]) -> None:
        """Test encoding without a secret key (insecure)."""
        with (
            patch.dict(os.environ, clear=True),
            patch(
                "datarobot_genai.core.utils.auth.get_authorization_context",
                return_value=auth_context,
            ),
        ):
            with pytest.warns(
                UserWarning,
                match="No secret key provided. Please make sure SESSION_SECRET_KEY is set.",
            ):
                token = AuthContextHeaderHandler(secret_key=None).encode()

        assert isinstance(token, str), (
            "Token should be a string even without secret key, to reduce dev friction."
        )
        assert token, "Token should not be empty even without secret key"

    def test_encode_with_explicit_context(
        self, handler: AuthContextHeaderHandler, auth_context: dict[str, Any]
    ) -> None:
        """Test encoding with explicitly provided authorization_context."""
        # Don't set up ContextVar - pass context explicitly
        token = handler.encode(authorization_context=auth_context)

        assert isinstance(token, str), "Token should be a string"
        assert token, "Token should not be empty"

        # Verify the token contains the explicit context
        decoded = handler.decode(token)
        assert decoded == auth_context

    def test_encode_explicit_context_priority_over_contextvar(
        self, handler: AuthContextHeaderHandler, auth_context: dict[str, Any]
    ) -> None:
        """Test that explicit authorization_context takes priority over ContextVar."""
        contextvar_context = {"user": {"id": "999", "name": "contextvar"}, "identities": []}
        explicit_context = auth_context

        # Set different context in ContextVar
        with patch(
            "datarobot_genai.core.utils.auth.get_authorization_context",
            return_value=contextvar_context,
        ):
            # Pass explicit context - should override ContextVar
            token = handler.encode(authorization_context=explicit_context)

            decoded = handler.decode(token)

            # Should use explicit context, not ContextVar
            assert decoded == explicit_context
            assert decoded != contextvar_context

    def test_encode_with_empty_explicit_context(self, handler: AuthContextHeaderHandler) -> None:
        """Test encoding with empty explicit authorization_context."""
        # set empty context in ContextVar, to ensure empty fallback
        set_authorization_context({})
        # encode with empty authorization_context
        token = handler.encode(authorization_context={})

        assert token is None, "Empty explicit context should return None"

    def test_encode_with_none_explicit_context_falls_back_to_contextvar(
        self, handler: AuthContextHeaderHandler, auth_context: dict[str, Any]
    ) -> None:
        """Test that None explicit context falls back to ContextVar."""
        # Set context in ContextVar
        with patch(
            "datarobot_genai.core.utils.auth.get_authorization_context", return_value=auth_context
        ):
            # Pass None explicitly - should fall back to ContextVar
            token = handler.encode(authorization_context=None)

            decoded = handler.decode(token)
            assert decoded == auth_context


class TestAuthContextHeaderHandlerDecode:
    """Tests for decoding JWT tokens into authorization context."""

    def test_decode_valid_token(
        self, handler: AuthContextHeaderHandler, auth_context: dict[str, Any], secret_key: str
    ) -> None:
        """Test decoding a valid JWT token."""
        token = jwt.encode(auth_context, secret_key, algorithm="HS256")
        decoded = handler.decode(token)

        assert isinstance(decoded, dict), "Decoded result should be a dict"
        assert decoded == auth_context, "Decoded context should match original"

    def test_decode_empty_token(self, handler: AuthContextHeaderHandler) -> None:
        """Test decoding an empty token."""
        decoded = handler.decode("")

        assert decoded is None, "Empty token should return None"

    def test_decode_with_invalid_signature(
        self, handler: AuthContextHeaderHandler, auth_context: dict[str, Any]
    ) -> None:
        """Test decoding a token with an invalid signature."""
        wrong_key = "wrong-secret-key"
        token = jwt.encode(auth_context, wrong_key, algorithm="HS256")
        decoded = handler.decode(token)

        assert decoded is None, "Token with invalid signature should return None"

    def test_decode_without_verification(
        self, handler_no_verification: AuthContextHeaderHandler, auth_context: dict[str, Any]
    ) -> None:
        """Test decoding without signature verification."""
        wrong_key = "wrong-secret-key"
        token = jwt.encode(auth_context, wrong_key, algorithm="HS256")
        decoded = handler_no_verification.decode(token)

        assert isinstance(decoded, dict), "Decoded result should be a dict"
        assert decoded == auth_context, "Decoded context should match original"

    def test_decode_malformed_token(self, handler: AuthContextHeaderHandler) -> None:
        """Test decoding a malformed token."""
        malformed_token = "invalid.malformed.token"
        decoded = handler.decode(malformed_token)

        assert decoded is None, "Malformed token should return None"


class TestAuthContextHeaderHandlerGetHeader:
    """Tests for getting authorization headers."""

    def test_get_header_with_context(
        self, handler: AuthContextHeaderHandler, auth_context: dict[str, Any]
    ) -> None:
        """Test getting header when authorization context is available."""
        with patch(
            "datarobot_genai.core.utils.auth.get_authorization_context", return_value=auth_context
        ):
            headers = handler.get_header()

            assert isinstance(headers, dict), "Headers should be a dict"
            assert AuthContextHeaderHandler.HEADER_NAME in headers, "Header key should be present"
            assert isinstance(headers[AuthContextHeaderHandler.HEADER_NAME], str), (
                "Header value should be a string"
            )

    def test_get_header_without_context(self, handler: AuthContextHeaderHandler) -> None:
        """Test getting header when no authorization context is available."""
        with patch("datarobot_genai.core.utils.auth.get_authorization_context", return_value=None):
            headers = handler.get_header()

            assert headers == {}, "Headers should be empty when no context is available"

    def test_get_header_with_explicit_context(
        self, handler: AuthContextHeaderHandler, auth_context: dict[str, Any]
    ) -> None:
        """Test getting header with explicitly provided authorization_context."""
        # Don't set up ContextVar - pass context explicitly
        headers = handler.get_header(authorization_context=auth_context)

        assert isinstance(headers, dict), "Headers should be a dict"
        assert AuthContextHeaderHandler.HEADER_NAME in headers, "Header key should be present"

        # Verify the token can be decoded and contains the explicit context
        token = headers[AuthContextHeaderHandler.HEADER_NAME]
        decoded = handler.decode(token)
        assert decoded == auth_context

    def test_get_header_explicit_context_priority_over_contextvar(
        self, handler: AuthContextHeaderHandler, auth_context: dict[str, Any]
    ) -> None:
        """Test that explicit authorization_context takes priority over ContextVar."""
        contextvar_context = {"user": {"id": "999", "name": "contextvar"}, "identities": []}
        explicit_context = auth_context

        # Set different context in ContextVar
        with patch(
            "datarobot_genai.core.utils.auth.get_authorization_context",
            return_value=contextvar_context,
        ):
            # Pass explicit context - should override ContextVar
            headers = handler.get_header(authorization_context=explicit_context)

            token = headers[AuthContextHeaderHandler.HEADER_NAME]
            decoded = handler.decode(token)

            # Should use explicit context, not ContextVar
            assert decoded == explicit_context
            assert decoded != contextvar_context

    def test_get_header_with_empty_explicit_context(
        self, handler: AuthContextHeaderHandler
    ) -> None:
        """Test getting header with empty explicit authorization_context."""
        headers = handler.get_header(authorization_context={})

        assert headers == {}, "Empty explicit context should return empty headers"


class TestAuthContextHeaderHandlerGetContext:
    """Tests for extracting authorization context from headers."""

    def test_get_context_from_valid_headers(
        self, handler: AuthContextHeaderHandler, auth_context: dict[str, Any], secret_key: str
    ) -> None:
        """Test extracting context from valid headers."""
        token = jwt.encode(auth_context, secret_key, algorithm="HS256")
        headers = {handler.header: token}

        ctx = handler.get_context(headers)

        assert isinstance(ctx, AuthCtx), "Result should be an AuthCtx instance"
        assert ctx.user.id == auth_context["user"]["id"]
        assert ctx.user.name == auth_context["user"]["name"]
        assert ctx.identities[0].id == auth_context["identities"][0]["id"]

    def test_get_context_from_headers_without_token(
        self, handler: AuthContextHeaderHandler
    ) -> None:
        """Test extracting context from headers without the auth token."""
        headers = {"Other-Header": "value"}

        ctx = handler.get_context(headers)

        assert ctx is None, "Context should be None when header is missing"

    def test_get_context_from_headers_with_invalid_token(
        self, handler: AuthContextHeaderHandler
    ) -> None:
        """Test extracting context from headers with an invalid token."""
        headers = {handler.header: "invalid.token"}

        ctx = handler.get_context(headers)

        assert ctx is None, "Context should be None when token is invalid"


class TestAuthContextHeaderHandlerRoundtrip:
    """Integration tests for encoding and decoding roundtrip."""

    def test_encode_decode_roundtrip(
        self, handler: AuthContextHeaderHandler, auth_context: dict[str, Any]
    ) -> None:
        """Test that encoding and then decoding returns the original context."""
        with patch(
            "datarobot_genai.core.utils.auth.get_authorization_context", return_value=auth_context
        ):
            token = handler.encode()
            decoded = handler.decode(token)

            assert decoded == auth_context, "Roundtrip should preserve the context"

    def test_full_header_workflow(
        self, handler: AuthContextHeaderHandler, auth_context: dict[str, Any]
    ) -> None:
        """Test the complete workflow: get_header -> get_context."""
        with patch(
            "datarobot_genai.core.utils.auth.get_authorization_context", return_value=auth_context
        ):
            headers = handler.get_header()
            ctx = handler.get_context(headers)

            assert isinstance(ctx, AuthCtx), "Result should be an AuthCtx instance"
            assert ctx.user.id == auth_context["user"]["id"]
            assert ctx.identities[0].id == auth_context["identities"][0]["id"]


class TestAsyncOAuthTokenProvider:
    """Tests for AsyncOAuthTokenProvider OAuth token management."""

    @pytest.fixture
    def auth_ctx_single_identity(self) -> AuthCtx:
        """AuthCtx with a single identity."""
        return AuthCtx(
            user=User(id="user123", email="user@example.com"),
            identities=[
                Identity(
                    id="id1",
                    provider_type="provider1",
                    type="oauth2",
                    provider_user_id="user1",
                    provider_identity_id="ea599021-acc3-490b-b2d7-a811ae1c9759",
                ),
            ],
        )

    @pytest.fixture
    def auth_ctx_multiple_identities(self) -> AuthCtx:
        """AuthCtx with multiple identities."""
        return AuthCtx(
            user=User(id="user123", email="user@example.com"),
            identities=[
                Identity(
                    id="id1",
                    provider_type="provider1",
                    type="oauth2",
                    provider_user_id="user1",
                    provider_identity_id="ea599021-acc3-490b-b2d7-a811ae1c9759",
                ),
                Identity(
                    id="id2",
                    provider_type="provider2",
                    type="oauth2",
                    provider_user_id="user2",
                    provider_identity_id="cc3f4426-9db1-4e77-bccb-72bcf7bc1ace",
                ),
            ],
        )

    @pytest.fixture
    def auth_ctx_datarobot_only(self) -> AuthCtx:
        """AuthCtx with only DataRobot identity."""
        return AuthCtx(
            user=User(id="user123", email="user@example.com"),
            identities=[
                Identity(
                    id="id1",
                    provider_type="datarobot_ext_email",
                    type="datarobot",
                    provider_user_id="user@example.com",
                    provider_identity_id=None,
                ),
            ],
        )

    @pytest.fixture
    def auth_ctx_obj(self) -> AuthCtx:
        """AuthCtx mimicking google OAuth scenario."""
        return AuthCtx(
            user=User(
                id="1",
                email="user@example.com",
                phone_number=None,
                name=None,
                given_name=None,
                family_name=None,
                profile_picture_url=None,
                metadata={},
            ),
            identities=[
                Identity(
                    id="1",
                    provider_type="datarobot_ext_email",
                    type="datarobot",
                    provider_user_id="user@example.com",
                    provider_identity_id=None,
                ),
                Identity(
                    id="2",
                    provider_type="google",
                    type="oauth2",
                    provider_user_id="user@example.com",
                    provider_identity_id="3d0edc76-95c6-4b75-9541-3fe17ccf068b",
                ),
            ],
        )

    @pytest.fixture
    def mock_token_data(self):
        """Mock token data returned by OAuth client."""
        token_data = AsyncMock()
        token_data.access_token = "test_access_token"
        return token_data

    def test_create_oauth_client_always_uses_datarobot(
        self, auth_ctx_single_identity: AuthCtx
    ) -> None:
        """Test that DataRobot OAuth client is always used regardless of provider type.

        DataRobot acts as the storage backend and token refresh utility for all OAuth providers.
        """
        with patch("datarobot_genai.core.utils.auth.DatarobotAsyncOAuthClient") as mock_dr:
            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_single_identity)
            mock_dr.assert_called_once()
            assert provider.oauth_client == mock_dr.return_value

    def test_get_identity_single(self, auth_ctx_single_identity: AuthCtx) -> None:
        """Test getting identity when only one OAuth identity is available."""
        with patch("datarobot_genai.core.utils.auth.DatarobotAsyncOAuthClient"):
            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_single_identity)
            identity = provider._get_identity(provider_type=None)
            assert identity.id == "id1"
            assert identity.provider_type == "provider1"

    def test_get_identity_by_provider_type(self, auth_ctx_multiple_identities: AuthCtx) -> None:
        """Test getting specific identity by provider_type."""
        with patch("datarobot_genai.core.utils.auth.DatarobotAsyncOAuthClient"):
            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_multiple_identities)
            identity = provider._get_identity(provider_type="provider2")
            assert identity.id == "id2"
            assert identity.provider_type == "provider2"

    def test_get_identity_multiple_without_provider_type(
        self, auth_ctx_multiple_identities: AuthCtx
    ) -> None:
        """Test that multiple identities without provider_type raises error."""
        with patch("datarobot_genai.core.utils.auth.DatarobotAsyncOAuthClient"):
            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_multiple_identities)
            with pytest.raises(
                ValueError,
                match="Multiple identities found. Please specify 'provider_type' parameter.",
            ):
                provider._get_identity(provider_type=None)

    def test_get_identity_no_match(self, auth_ctx_single_identity: AuthCtx) -> None:
        """Test error when provider_type doesn't match any identity."""
        with patch("datarobot_genai.core.utils.auth.DatarobotAsyncOAuthClient"):
            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_single_identity)
            with pytest.raises(ValueError, match="No identity found for provider 'unknown'"):
                provider._get_identity(provider_type="unknown")

    def test_get_identity_real_world_scenario(self, auth_ctx_obj: AuthCtx) -> None:
        """Test getting Google identity from real-world auth context.

        In real-world scenarios, users have a DataRobot identity (without provider_identity_id)
        and connected OAuth providers like Google (with provider_identity_id).
        """
        with patch("datarobot_genai.core.utils.auth.DatarobotAsyncOAuthClient"):
            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_obj)

            # Should get the Google identity when provider_type is specified
            identity = provider._get_identity(provider_type="google")
            assert identity.id == "2"
            assert identity.provider_type == "google"
            assert identity.provider_identity_id == "3d0edc76-95c6-4b75-9541-3fe17ccf068b"

            # Without provider_type, should get the only OAuth identity (Google)
            identity_default = provider._get_identity(provider_type=None)
            assert identity_default.id == "2"
            assert identity_default.provider_type == "google"

    def test_get_identity_filters_none_provider_identity_id(self) -> None:
        """Test that identities without provider_identity_id are filtered out."""
        auth_ctx = AuthCtx(
            user=User(id="user123", email="user@example.com"),
            identities=[
                Identity(
                    id="id1",
                    provider_type="provider1",
                    type="datarobot",
                    provider_user_id="user1",
                    provider_identity_id=None,  # Should be filtered out
                ),
                Identity(
                    id="id2",
                    provider_type="provider2",
                    type="oauth2",
                    provider_user_id="user2",
                    provider_identity_id="cc3f4426-9db1-4e77-bccb-72bcf7bc1ace",
                ),
            ],
        )
        with patch("datarobot_genai.core.utils.auth.DatarobotAsyncOAuthClient"):
            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx)
            identity = provider._get_identity(provider_type=None)
            assert identity.id == "id2"  # Only id2 has provider_identity_id

    def test_get_identity_no_valid_identities(self, auth_ctx_datarobot_only: AuthCtx) -> None:
        """Test error when no identities have provider_identity_id.

        This happens when a user is authenticated via DataRobot but hasn't connected
        any OAuth providers (Google, Microsoft, etc.).
        """
        with patch("datarobot_genai.core.utils.auth.DatarobotAsyncOAuthClient"):
            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_datarobot_only)
            with pytest.raises(ValueError, match="No identities found in authorization context."):
                provider._get_identity(provider_type=None)

    @pytest.mark.asyncio
    async def test_get_token_success(
        self, auth_ctx_single_identity: AuthCtx, mock_token_data
    ) -> None:
        """Test successful token retrieval using DataRobot OAuth client."""
        with patch(
            "datarobot_genai.core.utils.auth.DatarobotAsyncOAuthClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.refresh_access_token.return_value = mock_token_data
            mock_client_class.return_value = mock_client

            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_single_identity)
            token = await provider.get_token(auth_type=ToolAuth.OBO, provider_type="provider1")

            assert token == "test_access_token"
            mock_client.refresh_access_token.assert_called_once_with(
                identity_id="ea599021-acc3-490b-b2d7-a811ae1c9759"
            )

    @pytest.mark.asyncio
    async def test_get_token_with_multiple_identities(
        self, auth_ctx_multiple_identities: AuthCtx, mock_token_data
    ) -> None:
        """Test token retrieval with specific provider_type using DataRobot OAuth client."""
        with patch(
            "datarobot_genai.core.utils.auth.DatarobotAsyncOAuthClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.refresh_access_token.return_value = mock_token_data
            mock_client_class.return_value = mock_client

            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_multiple_identities)
            token = await provider.get_token(auth_type=ToolAuth.OBO, provider_type="provider2")

            assert token == "test_access_token"
            mock_client.refresh_access_token.assert_called_once_with(
                identity_id="cc3f4426-9db1-4e77-bccb-72bcf7bc1ace"
            )

    @pytest.mark.asyncio
    async def test_get_token_real_world_google(
        self, auth_ctx_obj: AuthCtx, mock_token_data
    ) -> None:
        """Test token retrieval for Google OAuth in real-world scenario.

        DataRobot manages the OAuth token refresh for Google and other providers.
        """
        with patch(
            "datarobot_genai.core.utils.auth.DatarobotAsyncOAuthClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.refresh_access_token.return_value = mock_token_data
            mock_client_class.return_value = mock_client

            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_obj)
            token = await provider.get_token(auth_type=ToolAuth.OBO, provider_type="google")

            assert token == "test_access_token"
            # Verify DataRobot client is used to refresh Google OAuth token
            mock_client.refresh_access_token.assert_called_once_with(
                identity_id="3d0edc76-95c6-4b75-9541-3fe17ccf068b"
            )

    @pytest.mark.asyncio
    async def test_get_token_unsupported_auth_type(self, auth_ctx_single_identity: AuthCtx) -> None:
        """Test error with unsupported auth type."""
        with patch("datarobot_genai.core.utils.auth.DatarobotAsyncOAuthClient"):
            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_single_identity)
            with pytest.raises(
                ValueError,
                match=r"Unsupported auth type: invalid\. Only OBO \(on-behalf-of\) is supported\.",
            ):
                await provider.get_token(auth_type="invalid", provider_type="provider1")

    @pytest.mark.asyncio
    async def test_get_token_no_oauth_providers_connected(
        self, auth_ctx_datarobot_only: AuthCtx
    ) -> None:
        """Test error when user has no OAuth providers connected.

        This occurs when a user is authenticated via DataRobot but hasn't connected
        any external OAuth providers like Google or Microsoft.
        """
        with patch("datarobot_genai.core.utils.auth.DatarobotAsyncOAuthClient"):
            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_datarobot_only)
            with pytest.raises(ValueError, match="No identities found in authorization context."):
                await provider.get_token(auth_type=ToolAuth.OBO)
