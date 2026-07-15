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

"""Tests for the ``execute_code`` function."""

import logging
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcputils import sandbox_mode
from datarobot_genai.drmcputils.exceptions import ToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind
from datarobot_genai.drmcputils.sandbox_mode import MCP_SANDBOX_DISABLED_ENV_VAR
from datarobot_genai.drtools.core.sandbox.base import SandboxError
from datarobot_genai.drtools.core.sandbox.base import SandboxResult
from datarobot_genai.drtools.core.sandbox.base import SandboxTimeout
from datarobot_genai.drtools.core.sandbox.utils import LOCAL_EXECUTION_IMAGE
from datarobot_genai.drtools.core.sandbox.utils import execute_code


@pytest.fixture
def dr_env() -> Iterator[None]:
    # `execute_code` derives credentials via the shared request/config helpers,
    # not os.environ — so stub the requesting user's token and the configured
    # endpoint rather than setting env vars.
    with (
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.get_datarobot_access_token",
            return_value="test-token",
        ),
        patch("datarobot_genai.drtools.core.sandbox.utils.get_credentials") as mock_creds,
    ):
        mock_creds.return_value.datarobot.datarobot_endpoint = "https://app.example.com/api/v2"
        yield


def _result(return_value: Any = None) -> SandboxResult:
    return SandboxResult(
        stdout="hi\n",
        stderr="",
        return_value=return_value,
        duration_s=0.01,
        exit_code=0,
    )


@pytest.fixture(autouse=True)
def _isolate_security_context_ff() -> Iterator[None]:
    # `_resolve_security_context` opens `request_user_dr_client()` (a context manager
    # that pings `/api/v2/version/` via client_configuration) and then evaluates the
    # FF. Stub both so the suite is hermetic and fast; security-context tests override
    # `is_enabled`. Also clear the per-(flag, principal) cache so results don't bleed.
    from datarobot_genai.drmcputils import feature_flags as _ff

    _ff._eval_cache.clear()
    with (
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.request_user_dr_client",
            return_value=MagicMock(),
        ),
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.FeatureFlag.is_enabled",
            return_value=False,
        ),
    ):
        yield


@pytest.mark.asyncio
async def test_execute_code_happy_path(dr_env: None) -> None:
    mock_run = AsyncMock(return_value=_result(return_value=42))
    with patch(
        "datarobot_genai.drtools.core.sandbox.utils.DataRobotWorkloadSandbox.run",
        new=mock_run,
    ):
        out = await execute_code("_return = 42")

    assert out["return_value"] == 42
    assert out["exit_code"] == 0
    assert out["image"] == (
        "datarobotdev/datarobot-user-models:public_dropin_environments"
        "_dr_mcp_execute_sandbox_minimal_latest"
    )
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_code_passes_security_context_when_ff_enabled(dr_env: None) -> None:
    from datarobot_genai.drtools.core.sandbox import utils as sandbox_utils

    mock_run = AsyncMock(return_value=_result())
    captured: dict[str, Any] = {}
    real_init = sandbox_utils.DataRobotWorkloadSandbox.__init__

    def _spy_init(self: Any, **kwargs: Any) -> None:
        captured["security_context"] = kwargs.get("security_context")
        real_init(self, **kwargs)

    with (
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.DataRobotWorkloadSandbox.run",
            new=mock_run,
        ),
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.DataRobotWorkloadSandbox.__init__",
            new=_spy_init,
        ),
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.FeatureFlag.is_enabled",
            return_value=True,
        ),
    ):
        await execute_code("_return = 1")

    assert captured["security_context"] is not None
    assert captured["security_context"].read_only_root_filesystem is True


@pytest.mark.asyncio
async def test_execute_code_omits_security_context_when_ff_disabled(dr_env: None) -> None:
    from datarobot_genai.drtools.core.sandbox import utils as sandbox_utils

    mock_run = AsyncMock(return_value=_result())
    captured: dict[str, Any] = {}
    real_init = sandbox_utils.DataRobotWorkloadSandbox.__init__

    def _spy_init(self: Any, **kwargs: Any) -> None:
        captured["security_context"] = kwargs.get("security_context")
        real_init(self, **kwargs)

    with (
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.DataRobotWorkloadSandbox.run",
            new=mock_run,
        ),
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.DataRobotWorkloadSandbox.__init__",
            new=_spy_init,
        ),
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.FeatureFlag.is_enabled",
            return_value=False,
        ),
    ):
        await execute_code("_return = 1")

    assert captured["security_context"] is None


@pytest.mark.asyncio
async def test_execute_code_omits_security_context_when_ff_check_raises(dr_env: None) -> None:
    from datarobot_genai.drtools.core.sandbox import utils as sandbox_utils

    mock_run = AsyncMock(return_value=_result())
    captured: dict[str, Any] = {}
    real_init = sandbox_utils.DataRobotWorkloadSandbox.__init__

    def _spy_init(self: Any, **kwargs: Any) -> None:
        captured["security_context"] = kwargs.get("security_context")
        real_init(self, **kwargs)

    with (
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.DataRobotWorkloadSandbox.run",
            new=mock_run,
        ),
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.DataRobotWorkloadSandbox.__init__",
            new=_spy_init,
        ),
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.FeatureFlag.is_enabled",
            side_effect=RuntimeError("DR client unavailable"),
        ),
    ):
        await execute_code("_return = 1")

    assert captured["security_context"] is None


@pytest.mark.asyncio
async def test_execute_code_derives_credentials_from_request(
    dr_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credentials come from the request helpers, never os.environ."""
    from datarobot_genai.drtools.core.sandbox import utils as sandbox_utils

    # Bogus env vars to prove they are ignored.
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://env-must-not-be-used.example/api/v2")
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "env-token-must-not-be-used")

    mock_run = AsyncMock(return_value=_result())
    captured: dict[str, Any] = {}
    real_init = sandbox_utils.DataRobotWorkloadSandbox.__init__

    def _spy_init(self: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        real_init(self, **kwargs)

    with (
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.DataRobotWorkloadSandbox.run",
            new=mock_run,
        ),
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.DataRobotWorkloadSandbox.__init__",
            new=_spy_init,
        ),
    ):
        await execute_code("_return = 1")

    assert captured["datarobot_api_token"] == "test-token"
    assert captured["datarobot_endpoint"] == "https://app.example.com/api/v2"


@pytest.mark.asyncio
async def test_execute_code_missing_token() -> None:
    """A missing/unauthorized token surfaces as ToolError(AUTHENTICATION)."""
    with patch(
        "datarobot_genai.drtools.core.sandbox.utils.get_datarobot_access_token",
        side_effect=ToolError("no token in headers", kind=ToolErrorKind.AUTHENTICATION),
    ):
        with pytest.raises(ToolError) as excinfo:
            await execute_code("_return = 1")

    assert excinfo.value.kind == ToolErrorKind.AUTHENTICATION


@pytest.mark.asyncio
async def test_execute_code_timeout_translates(dr_env: None) -> None:
    mock_run = AsyncMock(side_effect=SandboxTimeout("workload timed out"))
    with patch(
        "datarobot_genai.drtools.core.sandbox.utils.DataRobotWorkloadSandbox.run",
        new=mock_run,
    ):
        with pytest.raises(ToolError) as excinfo:
            await execute_code("import time; time.sleep(99)", timeout_s=0.1)

    assert excinfo.value.kind == ToolErrorKind.UPSTREAM
    assert "timed out" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_execute_code_sandbox_error_translates(dr_env: None) -> None:
    mock_run = AsyncMock(side_effect=SandboxError("workload failed"))
    with patch(
        "datarobot_genai.drtools.core.sandbox.utils.DataRobotWorkloadSandbox.run",
        new=mock_run,
    ):
        with pytest.raises(ToolError) as excinfo:
            await execute_code("raise RuntimeError('boom')")

    assert excinfo.value.kind == ToolErrorKind.UPSTREAM


# ---------------------------------------------------------------------------
# MCP_SANDBOX_DISABLED kill-switch
# ---------------------------------------------------------------------------


@pytest.fixture
def kill_switch_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MCP_SANDBOX_DISABLED_ENV_VAR, "true")
    monkeypatch.setattr(sandbox_mode, "_warning_emitted", False)


@pytest.mark.asyncio
async def test_execute_code_default_uses_workload_backend(dr_env: None) -> None:
    # GIVEN the kill-switch is NOT set (default posture)
    # WHEN execute_code runs
    mock_wrapper = MagicMock()
    mock_wrapper.return_value.run = AsyncMock(return_value=_result(return_value=1))
    with patch(
        "datarobot_genai.drtools.core.sandbox.utils.InstrumentedSandbox",
        new=mock_wrapper,
    ):
        await execute_code("_return = 1")
    # THEN the workload-api backend is selected
    inner = mock_wrapper.call_args.args[0]
    assert type(inner).__name__ == "DataRobotWorkloadSandbox"


@pytest.mark.asyncio
async def test_execute_code_kill_switch_runs_locally_without_dr_calls(
    kill_switch_on: None,
) -> None:
    # GIVEN MCP_SANDBOX_DISABLED=true and a DR API that would fail if touched
    with (
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.get_datarobot_access_token",
            side_effect=AssertionError("token derivation must not run when sandbox is disabled"),
        ),
        patch(
            "datarobot_genai.drtools.core.sandbox.utils.request_user_dr_client",
            side_effect=AssertionError("DR client must not be opened when sandbox is disabled"),
        ),
    ):
        # WHEN a real snippet executes
        out = await execute_code("print('local run')\n_return = 6 * 7")

    # THEN it runs locally on the MCP server process: no DR credential/FF
    # lookups, real output, and the local sentinel instead of a container image
    assert out["return_value"] == 42
    assert "local run" in out["stdout"]
    assert out["exit_code"] == 0
    assert out["image"] == LOCAL_EXECUTION_IMAGE


@pytest.mark.asyncio
async def test_execute_code_kill_switch_selects_local_backend(kill_switch_on: None) -> None:
    # GIVEN MCP_SANDBOX_DISABLED=true
    mock_wrapper = MagicMock()
    mock_wrapper.return_value.run = AsyncMock(return_value=_result(return_value=1))
    # WHEN execute_code runs
    with patch(
        "datarobot_genai.drtools.core.sandbox.utils.InstrumentedSandbox",
        new=mock_wrapper,
    ):
        await execute_code("_return = 1")
    # THEN the local backend is wrapped in the same instrumentation
    inner = mock_wrapper.call_args.args[0]
    assert type(inner).__name__ == "LocalProcessSandbox"


@pytest.mark.asyncio
async def test_execute_code_kill_switch_emits_warning(
    kill_switch_on: None, caplog: pytest.LogCaptureFixture
) -> None:
    # GIVEN MCP_SANDBOX_DISABLED=true
    # WHEN execute_code runs
    with caplog.at_level(logging.WARNING, logger=sandbox_mode.logger.name):
        await execute_code("_return = 1")
    # THEN a loud warning about disabled sandboxing is emitted
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("DISABLED" in m for m in messages)


@pytest.mark.asyncio
async def test_execute_code_kill_switch_local_error_translates(kill_switch_on: None) -> None:
    # GIVEN MCP_SANDBOX_DISABLED=true and a snippet that raises
    # WHEN execute_code runs
    with pytest.raises(ToolError) as excinfo:
        await execute_code("raise RuntimeError('local boom')")
    # THEN the failure surfaces as the same ToolError contract as the workload path
    assert excinfo.value.kind == ToolErrorKind.UPSTREAM


@pytest.mark.asyncio
async def test_execute_code_runs_through_instrumented_sandbox(dr_env: None) -> None:
    # The SLIs only fire if the workload sandbox is wrapped in InstrumentedSandbox.
    mock_wrapper = MagicMock()
    mock_wrapper.return_value.run = AsyncMock(return_value=_result())
    with patch(
        "datarobot_genai.drtools.core.sandbox.utils.InstrumentedSandbox",
        new=mock_wrapper,
    ):
        await execute_code("_return = None")

    mock_wrapper.assert_called_once()
    inner = mock_wrapper.call_args.args[0]
    assert type(inner).__name__ == "DataRobotWorkloadSandbox"
    mock_wrapper.return_value.run.assert_awaited_once()
