# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the MCP_SANDBOX_DISABLED kill-switch helper."""

import logging

import pytest

from datarobot_genai.drmcputils import sandbox_mode
from datarobot_genai.drmcputils.sandbox_mode import MCP_SANDBOX_DISABLED_ENV_VAR
from datarobot_genai.drmcputils.sandbox_mode import is_mcp_sandbox_disabled


@pytest.fixture(autouse=True)
def _reset_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep each test hermetic: env unset, warn-once latch cleared."""
    monkeypatch.delenv(MCP_SANDBOX_DISABLED_ENV_VAR, raising=False)
    monkeypatch.setattr(sandbox_mode, "_warning_emitted", False)


def test_sandbox_enabled_by_default() -> None:
    # GIVEN no MCP_SANDBOX_DISABLED in the environment
    # WHEN the kill-switch is evaluated
    # THEN sandboxing stays on (fail-closed default)
    assert is_mcp_sandbox_disabled() is False


@pytest.mark.parametrize("value", ["true", "True", "1", "yes"])
def test_kill_switch_disables_sandbox(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    # GIVEN MCP_SANDBOX_DISABLED set to a truthy value
    monkeypatch.setenv(MCP_SANDBOX_DISABLED_ENV_VAR, value)
    # WHEN the kill-switch is evaluated
    # THEN sandboxing is reported as disabled
    assert is_mcp_sandbox_disabled() is True


@pytest.mark.parametrize("value", ["false", "0", "no", ""])
def test_falsy_values_keep_sandbox_on(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    # GIVEN MCP_SANDBOX_DISABLED set to a falsy (or empty) value
    monkeypatch.setenv(MCP_SANDBOX_DISABLED_ENV_VAR, value)
    # WHEN the kill-switch is evaluated
    # THEN sandboxing stays on
    assert is_mcp_sandbox_disabled() is False


def test_unparseable_value_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN MCP_SANDBOX_DISABLED set to garbage that cannot parse as a bool
    monkeypatch.setenv(MCP_SANDBOX_DISABLED_ENV_VAR, "banana")
    # WHEN the kill-switch is evaluated
    # THEN the sandbox stays on (fail-closed) instead of raising
    assert is_mcp_sandbox_disabled() is False


def test_warning_emitted_once_when_disabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # GIVEN the kill-switch is on
    monkeypatch.setenv(MCP_SANDBOX_DISABLED_ENV_VAR, "true")
    # WHEN the kill-switch is evaluated multiple times
    with caplog.at_level(logging.WARNING, logger=sandbox_mode.logger.name):
        assert is_mcp_sandbox_disabled() is True
        assert is_mcp_sandbox_disabled() is True
    # THEN a loud warning is logged exactly once
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "DISABLED" in warnings[0].getMessage()
    assert "no isolation" in warnings[0].getMessage().lower()


def test_no_warning_when_sandbox_on(caplog: pytest.LogCaptureFixture) -> None:
    # GIVEN the kill-switch is off
    # WHEN the kill-switch is evaluated
    with caplog.at_level(logging.WARNING, logger=sandbox_mode.logger.name):
        assert is_mcp_sandbox_disabled() is False
    # THEN no warning is logged
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
