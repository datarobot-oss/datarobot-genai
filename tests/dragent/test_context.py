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

import jwt
import pytest
from nat.builder.context import ContextState
from nat.data_models.api_server import Request
from nat.runtime.user_metadata import RequestAttributes
from starlette.datastructures import Headers

from datarobot_genai.dragent.context import extract_authorization_from_context
from datarobot_genai.dragent.context import extract_datarobot_headers_from_context
from datarobot_genai.dragent.context import extract_headers_from_context


@pytest.fixture
def nat_context_set_headers():
    """Set NAT context metadata (e.g. request headers) for the test; reset on teardown."""
    context_state = ContextState.get()
    tokens = []

    def reset_context():
        while tokens:
            context_state._metadata.reset(tokens.pop())

    def set_headers(headers):
        """Set request headers in context. Pass a dict or None for no headers."""
        reset_context()
        attrs = RequestAttributes()
        attrs._request = Request(headers=Headers(headers) if headers is not None else None)
        tokens.append(context_state._metadata.set(attrs))

    yield set_headers
    reset_context()


@pytest.mark.parametrize(
    "headers,expected_headers",
    [
        (None, {}),
        ({}, {}),
        (
            {"X-DataRobot-Identity-Token": "identity-123", "X-Other-Header": "other-value"},
            {"x-datarobot-identity-token": "identity-123"},
        ),
        (
            {"X-Untrusted-Foo": "foo-val", "X-Other-Header": "other-value"},
            {"x-untrusted-foo": "foo-val"},
        ),
        (
            {
                "X-DataRobot-Identity-Token": "identity-123",
                "X-Untrusted-Foo": "foo-val",
                "X-Other-Header": "other-value",
            },
            {"x-datarobot-identity-token": "identity-123", "x-untrusted-foo": "foo-val"},
        ),
    ],
)
def test_extract_datarobot_headers_from_context(headers, expected_headers, nat_context_set_headers):
    nat_context_set_headers(headers)
    result = extract_datarobot_headers_from_context()
    assert result == expected_headers


@pytest.mark.parametrize(
    "headers,expected_authorization_context",
    [
        (None, None),
        ({}, None),
    ],
)
def test_extract_authorization_from_context_empty(
    headers, expected_authorization_context, nat_context_set_headers
):
    nat_context_set_headers(headers)
    result = extract_authorization_from_context()
    assert result == expected_authorization_context


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


def test_extract_authorization_from_context_with_auth_context_data(
    auth_context_data, secret_key, nat_context_set_headers, auth_token
):
    nat_context_set_headers({"X-DataRobot-Authorization-Context": auth_token})
    result = extract_authorization_from_context(secret_key=secret_key)
    assert result["user"]["id"] == auth_context_data["user"]["id"]
    assert result["identities"][0]["id"] == auth_context_data["identities"][0]["id"]
    assert result["metadata"]["endpoint"] == auth_context_data["metadata"]["endpoint"]


@pytest.mark.parametrize(
    "headers,headers_to_forward,expected_headers",
    [
        (None, ["Authorization"], {}),
        ({}, ["Authorization"], {}),
        (
            {"Authorization": "Bearer secret", "X-Request-Id": "req-123"},
            ["Authorization"],
            {"Authorization": "Bearer secret"},
        ),
        (
            {"Authorization": "Bearer secret"},
            ["Authorization", "X-Request-Id"],
            {"Authorization": "Bearer secret"},
        ),
    ],
)
def test_extract_headers_from_context(
    headers, headers_to_forward, expected_headers, nat_context_set_headers
):
    nat_context_set_headers(headers)
    result = extract_headers_from_context(headers_to_forward)
    assert result == expected_headers
