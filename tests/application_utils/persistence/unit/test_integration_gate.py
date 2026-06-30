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

"""Always-run guard for the live-integration env gate.

The integration suite is skipped unless three env vars are set; these tests pin the gate's
env-var wiring so a future typo in a variable name cannot silently disable the whole suite.
"""

from __future__ import annotations

from tests.application_utils.persistence.integration.test_persistence_orm_integration import (
    _live_enabled,
)

_FULL_ENV = {
    "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
    "DATAROBOT_API_TOKEN": "tok",
    "DR_MEMORY_LIVE_INTEGRATION": "1",
}


def test_gate_enabled_when_all_present() -> None:
    """GIVEN all three env vars set WHEN _live_enabled THEN True."""
    assert _live_enabled(_FULL_ENV) is True


def test_gate_accepts_documented_truthy_flag_values() -> None:
    """GIVEN each documented truthy flag value WHEN _live_enabled THEN True."""
    for flag in ("1", "true", "TRUE", "Yes"):
        assert _live_enabled({**_FULL_ENV, "DR_MEMORY_LIVE_INTEGRATION": flag}) is True


def test_gate_disabled_when_flag_unset() -> None:
    """GIVEN endpoint + token but no live flag WHEN _live_enabled THEN False."""
    env = {k: v for k, v in _FULL_ENV.items() if k != "DR_MEMORY_LIVE_INTEGRATION"}
    assert _live_enabled(env) is False


def test_gate_disabled_when_endpoint_missing() -> None:
    """GIVEN no DATAROBOT_ENDPOINT WHEN _live_enabled THEN False."""
    env = {k: v for k, v in _FULL_ENV.items() if k != "DATAROBOT_ENDPOINT"}
    assert _live_enabled(env) is False


def test_gate_disabled_when_token_missing() -> None:
    """GIVEN no DATAROBOT_API_TOKEN WHEN _live_enabled THEN False."""
    env = {k: v for k, v in _FULL_ENV.items() if k != "DATAROBOT_API_TOKEN"}
    assert _live_enabled(env) is False


def test_gate_disabled_for_unrecognized_flag_value() -> None:
    """GIVEN an unrecognised flag value WHEN _live_enabled THEN False."""
    assert _live_enabled({**_FULL_ENV, "DR_MEMORY_LIVE_INTEGRATION": "maybe"}) is False
