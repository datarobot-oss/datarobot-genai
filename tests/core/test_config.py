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

import pytest

from datarobot_genai.core.config import DEFAULT_MAX_HISTORY_MESSAGES
from datarobot_genai.core.config import get_max_history_messages_default


def test_get_max_history_messages_default_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATAROBOT_GENAI_MAX_HISTORY_MESSAGES", raising=False)
    assert get_max_history_messages_default() == DEFAULT_MAX_HISTORY_MESSAGES


def test_get_max_history_messages_default_env_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAROBOT_GENAI_MAX_HISTORY_MESSAGES", "nope")
    assert get_max_history_messages_default() == DEFAULT_MAX_HISTORY_MESSAGES


def test_get_max_history_messages_default_env_negative_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAROBOT_GENAI_MAX_HISTORY_MESSAGES", "-1")
    assert get_max_history_messages_default() == 0


def test_get_max_history_messages_default_env_zero_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAROBOT_GENAI_MAX_HISTORY_MESSAGES", "0")
    assert get_max_history_messages_default() == 0


def test_get_max_history_messages_default_env_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAROBOT_GENAI_MAX_HISTORY_MESSAGES", "7")
    assert get_max_history_messages_default() == 7
