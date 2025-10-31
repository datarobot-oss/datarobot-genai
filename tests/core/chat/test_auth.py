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

import pytest

from datarobot_genai.core.chat.auth import initialize_authorization_context


class _ContextRecorder:
    value: dict[str, Any] | None = None


@pytest.fixture(autouse=True)
def clear_ctx() -> None:
    _ContextRecorder.value = None


def _fake_set_authorization_context(value: dict[str, Any]) -> None:
    _ContextRecorder.value = value


@patch(
    "datarobot_genai.core.chat.auth.set_authorization_context",
    side_effect=_fake_set_authorization_context,
)
def test_initialize_authorization_context_sets_context(_: object) -> None:
    params = cast(dict[str, Any], {"authorization_context": {"foo": "bar"}})
    initialize_authorization_context(params)
    assert _ContextRecorder.value == {"foo": "bar"}


@patch(
    "datarobot_genai.core.chat.auth.set_authorization_context",
    side_effect=_fake_set_authorization_context,
)
def test_initialize_authorization_context_defaults_to_empty(_: object) -> None:
    params = cast(dict[str, Any], {})
    initialize_authorization_context(params)
    assert _ContextRecorder.value == {}
