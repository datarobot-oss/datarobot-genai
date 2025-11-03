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
from typing import cast
from unittest.mock import patch

import jwt
import pytest

from datarobot_genai.core.chat.auth import initialize_authorization_context


class _ContextRecorder:
    value: dict[str, Any] | None = None


@pytest.fixture(autouse=True)
def clear_ctx() -> None:
    _ContextRecorder.value = None


def _fake_set_authorization_context(value: dict[str, Any]) -> None:
    _ContextRecorder.value = value


@pytest.fixture
def mock_set_auth_context() -> Any:
    """Mock set_authorization_context to record calls."""
    with patch(
        "datarobot_genai.core.chat.auth.set_authorization_context",
        side_effect=_fake_set_authorization_context,
    ) as mock:
        yield mock


@pytest.fixture
def secret_key() -> str:
    """Return a test secret key for JWT signing."""
    return "test-secret-key"


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


def test_initialize_authorization_context_sets_context_from_params(mock_set_auth_context: Any) -> None:
    params = cast(dict[str, Any], {"authorization_context": {"foo": "bar"}})

    initialize_authorization_context(params)

    assert _ContextRecorder.value == {"foo": "bar"}


def test_initialize_authorization_context_defaults_to_empty(
    mock_set_auth_context: Any, empty_params: dict[str, Any]
) -> None:
    initialize_authorization_context(empty_params)

    assert _ContextRecorder.value == {}


def test_initialize_authorization_context_sets_context_from_headers(
    mock_set_auth_context: Any,
    empty_params: dict[str, Any],
    auth_token: str,
    secret_key: str,
    auth_context_data: dict[str, Any],
) -> None:
    headers = {"X-DataRobot-Authorization-Context": auth_token}

    initialize_authorization_context(empty_params, secret_key=secret_key, headers=headers)

    assert _ContextRecorder.value is not None
    assert _ContextRecorder.value["user"]["id"] == auth_context_data["user"]["id"]
    assert _ContextRecorder.value["user"]["name"] == auth_context_data["user"]["name"]
    assert _ContextRecorder.value["user"]["email"] == auth_context_data["user"]["email"]
    assert len(_ContextRecorder.value["identities"]) == 1
    assert _ContextRecorder.value["identities"][0]["id"] == auth_context_data["identities"][0]["id"]
    assert _ContextRecorder.value["identities"][0]["type"] == auth_context_data["identities"][0]["type"]
    assert _ContextRecorder.value["identities"][0]["provider_type"] == auth_context_data["identities"][0]["provider_type"]


def test_initialize_authorization_context_prefers_headers_over_params(
    mock_set_auth_context: Any,
    params_with_context: dict[str, Any],
    alt_auth_token: str,
    secret_key: str,
    alt_auth_context_data: dict[str, Any],
) -> None:
    headers = {"X-DataRobot-Authorization-Context": alt_auth_token}

    initialize_authorization_context(params_with_context, secret_key=secret_key, headers=headers)

    assert _ContextRecorder.value is not None
    assert _ContextRecorder.value["user"]["id"] == alt_auth_context_data["user"]["id"]
    assert _ContextRecorder.value["user"]["name"] == alt_auth_context_data["user"]["name"]
    assert _ContextRecorder.value["user"]["email"] == alt_auth_context_data["user"]["email"]
    assert "source" not in _ContextRecorder.value


def test_initialize_authorization_context_falls_back_to_params(
    mock_set_auth_context: Any,
    params_with_context: dict[str, Any],
    param_context: dict[str, Any],
    secret_key: str,
) -> None:
    headers = {}

    initialize_authorization_context(params_with_context, secret_key=secret_key, headers=headers)

    assert _ContextRecorder.value == param_context


def test_initialize_authorization_context_handles_invalid_jwt(
    mock_set_auth_context: Any,
    params_with_context: dict[str, Any],
    param_context: dict[str, Any],
    secret_key: str,
) -> None:
    headers = {"X-DataRobot-Authorization-Context": "invalid-token"}

    initialize_authorization_context(params_with_context, secret_key=secret_key, headers=headers)

    assert _ContextRecorder.value == param_context


def test_initialize_authorization_context_handles_wrong_secret(
    mock_set_auth_context: Any,
    params_with_context: dict[str, Any],
    param_context: dict[str, Any],
) -> None:
    invalid_auth_context = {"user_id": "123"}
    token = jwt.encode(invalid_auth_context, "correct-secret", algorithm="HS256")
    headers = {"X-DataRobot-Authorization-Context": token}

    initialize_authorization_context(params_with_context, secret_key="wrong-secret", headers=headers)

    assert _ContextRecorder.value == param_context

