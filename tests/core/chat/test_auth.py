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
import os
from typing import Any
from typing import cast
from unittest.mock import patch

import jwt
import pytest

from datarobot_genai.core.chat.auth import resolve_authorization_context


@pytest.fixture
def secret_key() -> str:
    """Return a test secret key for JWT signing."""
    return "test-secret-key"


@pytest.fixture(autouse=True)
def mock_secret_key(secret_key: str) -> None:
    """Mock the environment variable for JWT secret key."""
    with patch.dict(os.environ, {"SESSION_SECRET_KEY": secret_key}):
        yield


@pytest.fixture
def auth_context_data() -> dict[str, Any]:
    """Return sample authorization context data with required AuthCtx fields."""
    return {
        "user": {"id": "123", "name": "Test User", "email": "test@example.com"},
        "identities": [
            {"id": "id123", "type": "user", "provider_type": "github", "provider_user_id": "123"}
        ],
    }


@pytest.fixture
def alt_auth_context_data() -> dict[str, Any]:
    """Return alternative authorization context data for testing precedence."""
    return {
        "user": {"id": "456", "name": "Header User", "email": "header@example.com"},
        "identities": [
            {"id": "id456", "type": "user", "provider_type": "github", "provider_user_id": "456"}
        ],
    }


@pytest.fixture
def auth_token(auth_context_data: dict[str, Any], secret_key: str) -> str:
    """Generate a valid JWT token from auth context data."""
    return jwt.encode(auth_context_data, secret_key, algorithm="HS256")


@pytest.fixture
def alt_auth_token(alt_auth_context_data: dict[str, Any], secret_key: str) -> str:
    """Generate a valid JWT token from alternative auth context data."""
    return jwt.encode(alt_auth_context_data, secret_key, algorithm="HS256")


@pytest.fixture
def param_context() -> dict[str, Any]:
    """Return a simple param-based context for fallback tests."""
    return {"source": "params", "user_id": "789"}


@pytest.fixture
def empty_params() -> dict[str, Any]:
    """Return empty completion params."""
    return cast(dict[str, Any], {})


@pytest.fixture
def params_with_context(param_context: dict[str, Any]) -> dict[str, Any]:
    """Return completion params with authorization context."""
    return cast(dict[str, Any], {"authorization_context": param_context})


def test_resolve_authorization_context_sets_context_from_params() -> None:
    params = cast(dict[str, Any], {"authorization_context": {"foo": "bar"}})

    auth_ctx = resolve_authorization_context(params)

    assert auth_ctx == {"foo": "bar"}


def test_resolve_authorization_context_defaults_to_empty(empty_params: dict[str, Any]) -> None:
    auth_ctx = resolve_authorization_context(empty_params)

    assert auth_ctx == {}


def test_resolve_authorization_context_sets_context_from_headers(
    empty_params: dict[str, Any],
    auth_token: str,
    auth_context_data: dict[str, Any],
) -> None:
    headers = {"X-DataRobot-Authorization-Context": auth_token}

    auth_ctx = resolve_authorization_context(empty_params, headers=headers)

    assert auth_ctx is not None
    assert auth_ctx["user"]["id"] == auth_context_data["user"]["id"]
    assert auth_ctx["user"]["name"] == auth_context_data["user"]["name"]
    assert auth_ctx["user"]["email"] == auth_context_data["user"]["email"]
    assert len(auth_ctx["identities"]) == 1
    assert auth_ctx["identities"][0]["id"] == auth_context_data["identities"][0]["id"]
    assert auth_ctx["identities"][0]["type"] == auth_context_data["identities"][0]["type"]
    assert (
        auth_ctx["identities"][0]["provider_type"]
        == auth_context_data["identities"][0]["provider_type"]
    )


def test_resolve_authorization_context_prefers_headers_over_params(
    params_with_context: dict[str, Any],
    alt_auth_token: str,
    alt_auth_context_data: dict[str, Any],
) -> None:
    headers = {"X-DataRobot-Authorization-Context": alt_auth_token}

    auth_ctx = resolve_authorization_context(params_with_context, headers=headers)

    assert auth_ctx is not None
    assert auth_ctx["user"]["id"] == alt_auth_context_data["user"]["id"]
    assert auth_ctx["user"]["name"] == alt_auth_context_data["user"]["name"]
    assert auth_ctx["user"]["email"] == alt_auth_context_data["user"]["email"]
    assert "source" not in auth_ctx


def test_resolve_authorization_context_falls_back_to_params(
    params_with_context: dict[str, Any],
    param_context: dict[str, Any],
) -> None:
    headers = {}

    auth_ctx = resolve_authorization_context(params_with_context, headers=headers)

    assert auth_ctx == param_context


def test_resolve_authorization_context_handles_invalid_jwt(
    params_with_context: dict[str, Any],
    param_context: dict[str, Any],
) -> None:
    headers = {"X-DataRobot-Authorization-Context": "invalid-token"}

    auth_ctx = resolve_authorization_context(params_with_context, headers=headers)

    assert auth_ctx == param_context


def test_resolve_authorization_context_handles_wrong_secret(
    params_with_context: dict[str, Any],
    param_context: dict[str, Any],
) -> None:
    invalid_auth_context = {"user_id": "123"}
    token = jwt.encode(invalid_auth_context, "correct-secret", algorithm="HS256")
    headers = {"X-DataRobot-Authorization-Context": token}

    auth_ctx = resolve_authorization_context(
        params_with_context, secret_key="wrong-secret", headers=headers
    )

    assert auth_ctx == param_context


def test_resolve_authorization_context_with_lowercase_header(
    empty_params: dict[str, Any],
    auth_token: str,
    auth_context_data: dict[str, Any],
) -> None:
    """Test that lowercase header name is also recognized."""
    headers = {"x-datarobot-authorization-context": auth_token}

    auth_ctx = resolve_authorization_context(empty_params, headers=headers)

    assert auth_ctx is not None
    assert auth_ctx["user"]["id"] == auth_context_data["user"]["id"]


def test_resolve_authorization_context_with_empty_params_context(
    empty_params: dict[str, Any],
) -> None:
    """Test that empty authorization_context in params is handled correctly."""
    params = cast(dict[str, Any], {"authorization_context": {}})

    auth_ctx = resolve_authorization_context(params)

    assert auth_ctx == {}


def test_resolve_authorization_context_with_none_params_context(
    empty_params: dict[str, Any],
) -> None:
    """Test that None authorization_context in params is handled correctly."""
    params = cast(dict[str, Any], {"authorization_context": None})

    auth_ctx = resolve_authorization_context(params)

    assert auth_ctx == {}


def test_resolve_authorization_context_with_both_header_and_params_empty(
    empty_params: dict[str, Any],
) -> None:
    """Test that empty header and empty params result in empty context."""
    headers = {}

    auth_ctx = resolve_authorization_context(empty_params, headers=headers)

    assert auth_ctx == {}


def test_resolve_authorization_context_with_malformed_header_token(
    params_with_context: dict[str, Any],
    param_context: dict[str, Any],
) -> None:
    """Test handling of malformed JWT token in header."""
    headers = {"X-DataRobot-Authorization-Context": "invalid-jwt"}

    auth_ctx = resolve_authorization_context(params_with_context, headers=headers)

    # Should fall back to params
    assert auth_ctx == param_context


def test_resolve_authorization_context_with_missing_required_fields(
    empty_params: dict[str, Any],
    secret_key: str,
) -> None:
    """Test context with missing required fields for AuthCtx."""
    incomplete_context = {"user": {"id": "123"}}  # Missing name, email, identities

    token = jwt.encode(incomplete_context, secret_key, algorithm="HS256")
    headers = {"X-DataRobot-Authorization-Context": token}

    # Should return empty since AuthCtx validation will fail
    auth_ctx = resolve_authorization_context(empty_params, headers=headers)

    assert auth_ctx == {}


def test_resolve_authorization_context_integration_full_workflow(
    auth_context_data: dict[str, Any],
    secret_key: str,
) -> None:
    """Integration test: full workflow from encoding to resolution."""
    # Step 1: Create a token (simulating agent sending context)
    token = jwt.encode(auth_context_data, secret_key, algorithm="HS256")

    # Step 2: Simulate receiving the token in headers
    headers = {"X-DataRobot-Authorization-Context": token}
    params = cast(dict[str, Any], {})

    # Step 3: Resolve the context
    auth_ctx = resolve_authorization_context(params, headers=headers)

    # Step 4: Verify the context was fully preserved
    assert auth_ctx == auth_context_data
    assert auth_ctx["user"]["id"] == "123"
    assert auth_ctx["user"]["name"] == "Test User"
    assert len(auth_ctx["identities"]) == 1


def test_resolve_authorization_context_preserves_all_identity_fields(
    empty_params: dict[str, Any],
    secret_key: str,
) -> None:
    """Test that all identity fields are preserved through resolution."""
    context_with_full_identity = {
        "user": {"id": "123", "name": "Test", "email": "test@example.com"},
        "identities": [
            {
                "id": "id123",
                "type": "user",
                "provider_type": "github",
                "provider_user_id": "github123",
                "provider_identity_id": "provider456",
                "metadata": {"extra": "data"},
            }
        ],
    }

    token = jwt.encode(context_with_full_identity, secret_key, algorithm="HS256")
    headers = {"X-DataRobot-Authorization-Context": token}

    auth_ctx = resolve_authorization_context(empty_params, headers=headers)

    assert auth_ctx["identities"][0]["provider_identity_id"] == "provider456"
    assert auth_ctx["identities"][0]["metadata"]["extra"] == "data"
