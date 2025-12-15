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

import aiohttp
import jwt
import pytest
from aioresponses import aioresponses
from datarobot.auth.exceptions import OAuthProviderNotFound
from datarobot.auth.exceptions import OAuthValidationErr
from datarobot.auth.identity import Identity
from datarobot.auth.session import AuthCtx
from datarobot.auth.users import User
from datarobot.models.genai.agent.auth import ToolAuth
from datarobot.models.genai.agent.auth import set_authorization_context

from datarobot_genai.core.utils.auth import AsyncOAuthTokenProvider
from datarobot_genai.core.utils.auth import AuthContextHeaderHandler
from datarobot_genai.core.utils.auth import AuthlibTokenRetriever
from datarobot_genai.core.utils.auth import DatarobotTokenRetriever
from datarobot_genai.core.utils.auth import OAuthConfig
from datarobot_genai.core.utils.auth import create_token_retriever


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


@pytest.fixture
def mock_datarobot_client():
    # mock dr client, to avoid external calls
    with patch("datarobot_genai.core.utils.auth.DatarobotOAuthClient") as m:
        yield m


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
    def auth_ctx_obj_datarobot_impl(self) -> AuthCtx:
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
    def auth_ctx_obj_authlib_impl(self) -> AuthCtx:
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
                    provider_identity_id=None,
                ),
            ],
            metadata={
                "oauth_implementation": "authlib",
                "application_endpoint": "https://app.example.com/api/v1",
            },
        )

    @pytest.fixture
    def mock_token_data(self):
        """Mock token data returned by OAuth client."""
        token_data = AsyncMock()
        token_data.access_token = "test_access_token"
        return token_data

    def test_oauth_uses_datarobot(self, auth_ctx_obj_datarobot_impl: AuthCtx) -> None:
        with patch("datarobot_genai.core.utils.auth.DatarobotTokenRetriever") as mock_retriever:
            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_obj_datarobot_impl)
            mock_retriever.assert_called_once()
            assert provider._retriever == mock_retriever.return_value

    def test_oauth_uses_authlib(self, auth_ctx_obj_authlib_impl: AuthCtx) -> None:
        with patch("datarobot_genai.core.utils.auth.AuthlibTokenRetriever") as mock_retriever:
            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_obj_authlib_impl)
            mock_retriever.assert_called_once()
            assert provider._retriever == mock_retriever.return_value

    def test_get_identity_single(
        self, mock_datarobot_client, auth_ctx_single_identity: AuthCtx
    ) -> None:
        """Test getting single identity."""
        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_single_identity)
        identity = provider._get_identity(provider_type=None)
        assert identity.id == "id1"
        assert identity.provider_type == "provider1"

    def test_get_identity_by_provider_type(
        self, mock_datarobot_client, auth_ctx_multiple_identities: AuthCtx
    ) -> None:
        """Test getting identity by provider type."""
        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_multiple_identities)

        identity = provider._get_identity(provider_type="provider2")
        assert identity.id == "id2"
        assert identity.provider_type == "provider2"

        identity = provider._get_identity(provider_type="provider1")
        assert identity.id == "id1"
        assert identity.provider_type == "provider1"

    def test_get_identity_multiple_without_provider_type_raises(
        self, mock_datarobot_client, auth_ctx_multiple_identities: AuthCtx
    ) -> None:
        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_multiple_identities)
        with pytest.raises(
            OAuthValidationErr,
            match="Multiple OAuth providers found. Specify 'provider_type' parameter.",
        ):
            provider._get_identity(provider_type=None)

    def test_get_identity_unknown_provider_raises(
        self, mock_datarobot_client, auth_ctx_single_identity: AuthCtx
    ) -> None:
        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_single_identity)
        with pytest.raises(ValueError, match="No identity found for provider 'unknown'"):
            provider._get_identity(provider_type="unknown")

    def test_get_identity_google_oauth(
        self, mock_datarobot_client, auth_ctx_obj_datarobot_impl: AuthCtx
    ) -> None:
        """Test getting Google OAuth identity."""
        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_obj_datarobot_impl)

        # Should get the Google identity when provider_type is specified
        identity = provider._get_identity(provider_type="google")
        assert identity.id == "2"
        assert identity.provider_type == "google"
        assert identity.provider_identity_id == "3d0edc76-95c6-4b75-9541-3fe17ccf068b"

        # Without provider_type, should get the only OAuth identity (Google)
        identity_default = provider._get_identity(provider_type=None)
        assert identity_default.id == "2"
        assert identity_default.provider_type == "google"

    def test_get_identity_filters_by_provider_identity_id(
        self, mock_datarobot_client, auth_ctx_multiple_identities: AuthCtx
    ) -> None:
        """Test that only identities with provider_identity_id are returned."""
        # Create auth context where only id2 has provider_identity_id
        auth_ctx = AuthCtx(
            user=User(id="user123", email="user@example.com"),
            identities=[
                Identity(
                    id="id1",
                    provider_type="provider1",
                    type="oauth2",
                    provider_user_id="user1",
                    provider_identity_id=None,  # No provider_identity_id
                ),
                Identity(
                    id="id2",
                    provider_type="provider2",
                    type="oauth2",
                    provider_user_id="user2",
                    provider_identity_id="has-provider-id",
                ),
            ],
        )

        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx)
        identity = provider._get_identity(provider_type=None)
        assert identity.id == "id2"  # Only id2 has provider_identity_id

    def test_get_identity_no_valid_identities_raises(
        self, mock_datarobot_client, auth_ctx_datarobot_only: AuthCtx
    ) -> None:
        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_datarobot_only)
        with pytest.raises(OAuthProviderNotFound, match="No OAuth provider found."):
            provider._get_identity(provider_type=None)

    @pytest.mark.asyncio
    async def test_get_token_single_identity(
        self, mock_datarobot_client, auth_ctx_single_identity: AuthCtx, mock_token_data: AsyncMock
    ) -> None:
        """Test getting token with single identity."""
        mock_client = mock_datarobot_client.return_value
        mock_client.refresh_access_token = AsyncMock(return_value=mock_token_data)

        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_single_identity)
        token = await provider.get_token(auth_type=ToolAuth.OBO, provider_type="provider1")

        assert token == "test_access_token"
        args, kwargs = mock_client.refresh_access_token.call_args
        assert kwargs["provider_id"] == "ea599021-acc3-490b-b2d7-a811ae1c9759"

    @pytest.mark.asyncio
    async def test_get_token_multiple_identities(
        self,
        mock_datarobot_client,
        auth_ctx_multiple_identities: AuthCtx,
        mock_token_data: AsyncMock,
    ) -> None:
        """Test getting token with multiple identities."""
        mock_client = mock_datarobot_client.return_value
        mock_client.refresh_access_token = AsyncMock(return_value=mock_token_data)

        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_multiple_identities)
        token = await provider.get_token(auth_type=ToolAuth.OBO, provider_type="provider2")

        assert token == "test_access_token"
        args, kwargs = mock_client.refresh_access_token.call_args
        assert kwargs["provider_id"] == "cc3f4426-9db1-4e77-bccb-72bcf7bc1ace"

    @pytest.mark.asyncio
    async def test_get_token_google_oauth(
        self,
        mock_datarobot_client,
        auth_ctx_obj_datarobot_impl: AuthCtx,
        mock_token_data: AsyncMock,
    ) -> None:
        mock_client = mock_datarobot_client.return_value
        mock_client.refresh_access_token = AsyncMock(return_value=mock_token_data)

        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_obj_datarobot_impl)
        token = await provider.get_token(auth_type=ToolAuth.OBO, provider_type="google")

        assert token == "test_access_token"
        args, kwargs = mock_client.refresh_access_token.call_args
        assert kwargs["provider_id"] == "3d0edc76-95c6-4b75-9541-3fe17ccf068b"

    @pytest.mark.asyncio
    async def test_get_token_unsupported_auth_type_raises(
        self, mock_datarobot_client, auth_ctx_single_identity: AuthCtx
    ) -> None:
        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_single_identity)
        fake_auth_type = "FAKE_AUTH_TYPE"
        with pytest.raises(ValueError, match=r"Unsupported auth type"):
            await provider.get_token(
                auth_type=fake_auth_type,
                provider_type="provider1",  # type: ignore
            )

    @pytest.mark.asyncio
    async def test_get_token_no_valid_identities_raises(
        self, mock_datarobot_client, auth_ctx_datarobot_only: AuthCtx
    ) -> None:
        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_datarobot_only)
        with pytest.raises(OAuthProviderNotFound, match="No OAuth provider found."):
            await provider.get_token(auth_type=ToolAuth.OBO)


class TestOAuthConfig:
    """Tests for OAuthConfig class."""

    def test_from_auth_ctx_with_datarobot_implementation(self) -> None:
        """Test extracting OAuth config with explicit datarobot implementation."""
        auth_ctx = AuthCtx(
            user=User(id="user123", email="user@example.com"),
            identities=[],
            metadata={"oauth_implementation": "datarobot"},
        )

        config = OAuthConfig.from_auth_ctx(auth_ctx)

        assert config.implementation == "datarobot"
        assert config.application_endpoint is None

    def test_from_auth_ctx_with_authlib_implementation(self) -> None:
        """Test extracting OAuth config with authlib implementation."""
        auth_ctx = AuthCtx(
            user=User(id="user123", email="user@example.com"),
            identities=[],
            metadata={
                "oauth_implementation": "authlib",
                "application_endpoint": "https://app.example.com",
            },
        )

        config = OAuthConfig.from_auth_ctx(auth_ctx)

        assert config.implementation == "authlib"
        assert config.application_endpoint == "https://app.example.com"

    def test_from_auth_ctx_defaults_to_datarobot(self) -> None:
        """Test that default implementation is datarobot when not specified."""
        auth_ctx = AuthCtx(
            user=User(id="user123", email="user@example.com"),
            identities=[],
            metadata={},
        )

        config = OAuthConfig.from_auth_ctx(auth_ctx)

        assert config.implementation == "datarobot"
        assert config.application_endpoint is None

    def test_from_auth_ctx_with_no_metadata(self) -> None:
        """Test extracting OAuth config when metadata is None."""
        auth_ctx = AuthCtx(
            user=User(id="user123", email="user@example.com"),
            identities=[],
            metadata=None,
        )

        config = OAuthConfig.from_auth_ctx(auth_ctx)

        assert config.implementation == "datarobot"
        assert config.application_endpoint is None


class TestCreateTokenRetriever:
    """Tests for create_token_retriever factory function."""

    def test_create_datarobot_retriever(self, mock_datarobot_client) -> None:
        """Test creating DataRobot token retriever."""
        config = OAuthConfig(implementation="datarobot")
        retriever = create_token_retriever(config)
        assert isinstance(retriever, DatarobotTokenRetriever)

    def test_create_authlib_retriever(self) -> None:
        """Test creating Authlib token retriever."""
        config = OAuthConfig(
            implementation="authlib", application_endpoint="https://app.example.com"
        )

        retriever = create_token_retriever(config)

        assert isinstance(retriever, AuthlibTokenRetriever)
        assert retriever.application_endpoint == "https://app.example.com"

    def test_create_authlib_retriever_without_endpoint_raises_error(self) -> None:
        """Test that creating Authlib retriever without endpoint raises ValueError."""
        config = OAuthConfig(implementation="authlib", application_endpoint=None)

        with pytest.raises(
            ValueError,
            match="Required 'application_endpoint' not found in metadata.",
        ):
            create_token_retriever(config)

    def test_create_retriever_with_unsupported_implementation(self) -> None:
        """Test that unsupported implementation raises ValueError."""
        config = OAuthConfig(implementation="unsupported")

        with pytest.raises(
            ValueError,
            match="Unsupported OAuth implementation: 'unsupported'.",
        ):
            create_token_retriever(config)


class TestAuthlibTokenRetriever:
    """Tests for AuthlibTokenRetriever class."""

    def test_init_with_valid_endpoint(self) -> None:
        """Test initializing with a valid endpoint."""
        retriever = AuthlibTokenRetriever("https://app.example.com")

        assert retriever.application_endpoint == "https://app.example.com"

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from endpoint."""
        retriever = AuthlibTokenRetriever("https://app.example.com/")

        assert retriever.application_endpoint == "https://app.example.com"

    def test_init_without_endpoint_raises_error(self) -> None:
        """Test that initializing without endpoint raises ValueError."""
        with pytest.raises(
            ValueError, match="AuthlibTokenRetriever requires 'application_endpoint'"
        ):
            AuthlibTokenRetriever("")

    @pytest.mark.asyncio
    async def test_refresh_access_token_success(self) -> None:
        """Test successfully refreshing an access token."""
        retriever = AuthlibTokenRetriever("https://app.example.com")
        identity = Identity(
            id="test-identity-123",
            provider_type="google",
            type="oauth2",
            provider_user_id="user@example.com",
            provider_identity_id=None,
        )

        mock_response_data = {
            "access_token": "new_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with (
            patch.dict(os.environ, {"DATAROBOT_API_TOKEN": "test-api-token"}),
            patch("aiohttp.ClientSession.post") as mock_post,
        ):
            mock_response = AsyncMock()
            mock_response.raise_for_status = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_post.return_value.__aenter__.return_value = mock_response

            token = await retriever.refresh_access_token(identity)

            assert token.access_token == "new_access_token"
            assert token.token_type == "Bearer"
            assert token.expires_in == 3600

            # Verify the request was made correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "https://app.example.com/oauth/token/"
            assert call_args[1]["headers"] == {"Authorization": "Bearer test-api-token"}
            assert call_args[1]["json"] == {"identity_id": identity.id}

    @pytest.mark.asyncio
    async def test_refresh_access_token_without_api_token_raises_error(self) -> None:
        """Test that missing DATAROBOT_API_TOKEN raises ValueError."""
        retriever = AuthlibTokenRetriever("https://app.example.com")
        identity = Identity(
            id="test-identity-123",
            provider_type="google",
            type="oauth2",
            provider_user_id="user@example.com",
            provider_identity_id=None,
        )

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError,
                match="DATAROBOT_API_TOKEN environment variable is required but not set.",
            ):
                await retriever.refresh_access_token(identity)

    @pytest.mark.asyncio
    async def test_refresh_access_token_http_error(self) -> None:
        """Test handling HTTP errors during token refresh."""
        retriever = AuthlibTokenRetriever("https://app.example.com")
        identity = Identity(
            id="test-identity-123",
            provider_type="google",
            type="oauth2",
            provider_user_id="user@example.com",
            provider_identity_id=None,
        )

        with (
            patch.dict(os.environ, {"DATAROBOT_API_TOKEN": "test-api-token"}),
            aioresponses() as m,
        ):
            # Mock the POST request to raise an HTTP error
            m.post("https://app.example.com/oauth/token/", status=500, body="Internal Server Error")

            with pytest.raises(aiohttp.ClientError):
                await retriever.refresh_access_token(identity)

    @pytest.mark.asyncio
    async def test_refresh_access_token_with_responses_mock(self) -> None:
        """Test token refresh with aioresponses mocking the authlib endpoint."""
        retriever = AuthlibTokenRetriever("https://app.example.com/api/v1/")
        identity = Identity(
            id="test-identity-123",
            provider_type="google",
            type="oauth2",
            provider_user_id="user@example.com",
            provider_identity_id=None,
        )

        mock_response_data = {
            "access_token": "access-token-value",
            "token_type": None,
            "expires_in": None,
            "expires_at": None,
            "refresh_token": None,
            "id_token": None,
            "scope": None,
        }

        with (
            patch.dict(os.environ, {"DATAROBOT_API_TOKEN": "test-api-token"}),
            aioresponses() as m,
        ):
            # Mock the POST request with 200 OK response
            m.post(
                "https://app.example.com/api/v1/oauth/token/",
                status=200,
                payload=mock_response_data,
            )

            token = await retriever.refresh_access_token(identity)

            assert token.access_token == "access-token-value"
            assert token.token_type is None
            assert token.expires_in is None


class TestAsyncOAuthTokenProviderWithAuthlib:
    """Tests for AsyncOAuthTokenProvider using Authlib implementation."""

    @pytest.fixture
    def auth_ctx_authlib(self) -> AuthCtx:
        """AuthCtx configured to use Authlib implementation."""
        return AuthCtx(
            user=User(id="user123", email="user@example.com"),
            identities=[
                Identity(
                    id="id1",
                    provider_type="google",
                    type="oauth2",
                    provider_user_id="user@example.com",
                    provider_identity_id=None,  # Authlib uses None
                )
            ],
            metadata={
                "oauth_implementation": "authlib",
                "application_endpoint": "https://app.example.com",
            },
        )

    def test_init_with_authlib_implementation(self, auth_ctx_authlib: AuthCtx) -> None:
        """Test initializing provider with Authlib implementation."""
        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_authlib)

        assert isinstance(provider._retriever, AuthlibTokenRetriever)
        assert provider._retriever.application_endpoint == "https://app.example.com"

    def test_init_with_authlib_without_endpoint_raises_error(self) -> None:
        """Test that Authlib without endpoint raises ValueError."""
        auth_ctx = AuthCtx(
            user=User(id="user123", email="user@example.com"),
            identities=[],
            metadata={"oauth_implementation": "authlib"},
        )

        with pytest.raises(
            ValueError,
            match="Required 'application_endpoint' not found in metadata.",
        ):
            AsyncOAuthTokenProvider(auth_ctx=auth_ctx)

    @pytest.mark.asyncio
    async def test_get_token_with_authlib(self, auth_ctx_authlib: AuthCtx) -> None:
        """Test getting token using Authlib implementation."""
        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx_authlib)

        mock_token_data = AsyncMock()
        mock_token_data.access_token = "authlib_access_token"

        with patch.object(
            provider._retriever, "refresh_access_token", return_value=mock_token_data
        ):
            token = await provider.get_token(auth_type=ToolAuth.OBO, provider_type="google")

            assert token == "authlib_access_token"
            # Verify the identity object was passed (not just the ID)
            call_args = provider._retriever.refresh_access_token.call_args
            identity = call_args[0][0]
            assert identity.id == "id1"
            assert identity.provider_type == "google"

    def test_init_with_datarobot_implementation_explicitly(self, mock_datarobot_client) -> None:
        """Test initializing provider with explicit DataRobot implementation."""
        auth_ctx = AuthCtx(
            user=User(id="user123", email="user@example.com"),
            identities=[
                Identity(
                    id="id1",
                    provider_type="google",
                    type="oauth2",
                    provider_user_id="user@example.com",
                    provider_identity_id="test-id",
                )
            ],
            metadata={"oauth_implementation": "datarobot"},
        )

        with patch.dict(os.environ, {"DATAROBOT_ENDPOINT": "https://app.example.com"}):
            provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx)
        assert isinstance(provider._retriever, DatarobotTokenRetriever)

    @pytest.mark.asyncio
    async def test_get_token_with_multiple_identities_authlib(self) -> None:
        """Test getting token with multiple identities using Authlib."""
        auth_ctx = AuthCtx(
            user=User(id="user123", email="user@example.com"),
            identities=[
                Identity(
                    id="id1",
                    provider_type="google",
                    type="oauth2",
                    provider_user_id="user1@example.com",
                    provider_identity_id=None,  # Authlib uses None
                ),
                Identity(
                    id="id2",
                    provider_type="microsoft",
                    type="oauth2",
                    provider_user_id="user2@example.com",
                    provider_identity_id=None,  # Authlib uses None
                ),
            ],
            metadata={
                "oauth_implementation": "authlib",
                "application_endpoint": "https://app.example.com",
            },
        )

        provider = AsyncOAuthTokenProvider(auth_ctx=auth_ctx)

        mock_token_data = AsyncMock()
        mock_token_data.access_token = "microsoft_token"

        with patch.object(
            provider._retriever, "refresh_access_token", return_value=mock_token_data
        ):
            token = await provider.get_token(auth_type=ToolAuth.OBO, provider_type="microsoft")

            assert token == "microsoft_token"
            # Verify the identity object was passed
            call_args = provider._retriever.refresh_access_token.call_args
            identity = call_args[0][0]
            assert identity.id == "id2"
            assert identity.provider_type == "microsoft"
