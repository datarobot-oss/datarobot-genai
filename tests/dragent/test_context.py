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

import pytest
from nat.builder.context import ContextState
from nat.data_models.api_server import Request
from nat.runtime.user_metadata import RequestAttributes
from starlette.datastructures import Headers

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
