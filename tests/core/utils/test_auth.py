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
from unittest.mock import patch

import jwt
import pytest
from datarobot.auth.session import AuthCtx

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
