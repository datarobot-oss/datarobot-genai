# Copyright 2026 DataRobot, Inc. and its affiliates.
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

"""Shared fixtures for the dragent test packages."""

from collections.abc import Iterator

import pytest

from datarobot_genai.dragent.inbound_token import _authorization_carries_idp_token


@pytest.fixture(autouse=True)
def _reset_inbound_token_settings() -> Iterator[None]:
    """Drop the cached ``DRAGENT_ALLOW_IDP_TOKEN_IN_AUTHORIZATION`` read around every test.

    ``find_idp_token`` resolves the settings chain once and caches it, so without this the
    first test to touch it would pin the value for the rest of the session and the opt-in
    tests would pass or fail depending on collection order.
    """
    _authorization_carries_idp_token.cache_clear()
    yield
    _authorization_carries_idp_token.cache_clear()


@pytest.fixture
def authorization_carries_idp_token(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Opt in to reading the IdP token from ``authorization``, as a local run with no gateway."""
    monkeypatch.setenv("DRAGENT_ALLOW_IDP_TOKEN_IN_AUTHORIZATION", "true")
    _authorization_carries_idp_token.cache_clear()
    yield
    _authorization_carries_idp_token.cache_clear()
