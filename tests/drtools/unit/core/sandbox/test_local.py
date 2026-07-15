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

"""Tests for :class:`LocalProcessSandbox` — real subprocess execution."""

import pytest

from datarobot_genai.drtools.core.sandbox.base import Sandbox
from datarobot_genai.drtools.core.sandbox.base import SandboxError
from datarobot_genai.drtools.core.sandbox.base import SandboxTimeout
from datarobot_genai.drtools.core.sandbox.local import LocalProcessSandbox
from datarobot_genai.drtools.core.sandbox.protocol import RESULT_MARKER


def test_local_sandbox_satisfies_protocol() -> None:
    # GIVEN the LocalProcessSandbox class
    # THEN it satisfies the Sandbox protocol
    assert isinstance(LocalProcessSandbox(), Sandbox)


@pytest.mark.asyncio
async def test_run_returns_stdout_and_return_value() -> None:
    # GIVEN a snippet that prints and assigns `_return` from `inputs`
    sandbox = LocalProcessSandbox()
    code = "print('hello from local')\n_return = inputs['x'] * 2"
    # WHEN it runs in the local sandbox
    result = await sandbox.run(code, inputs={"x": 21}, timeout_s=30.0)
    # THEN stdout, return value, and exit code round-trip
    assert "hello from local" in result.stdout
    assert result.return_value == 42
    assert result.exit_code == 0
    assert result.duration_s > 0
    # AND the wire marker is stripped from user-visible stdout
    assert RESULT_MARKER not in result.stdout


@pytest.mark.asyncio
async def test_run_without_inputs_or_return() -> None:
    # GIVEN a snippet with no inputs and no `_return`
    sandbox = LocalProcessSandbox()
    # WHEN it runs
    result = await sandbox.run("x = 1 + 1")
    # THEN it succeeds with a None return value
    assert result.return_value is None
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_run_can_import_installed_packages() -> None:
    # GIVEN a snippet that imports a real third-party package (polars), as the
    # panel transform preamble does — this is the capability pydantic-monty
    # lacks and the reason the local backend is a plain subprocess.
    sandbox = LocalProcessSandbox()
    code = (
        "import polars as pl\n"
        "df = pl.DataFrame(inputs['rows'])\n"
        "_return = df.filter(pl.col('a') > 1).to_dicts()"
    )
    # WHEN it runs
    result = await sandbox.run(code, inputs={"rows": [{"a": 1}, {"a": 2}]})
    # THEN the polars-backed transform works
    assert result.return_value == [{"a": 2}]


@pytest.mark.asyncio
async def test_user_error_raises_sandbox_error_with_traceback() -> None:
    # GIVEN a snippet that raises
    sandbox = LocalProcessSandbox()
    # WHEN it runs
    with pytest.raises(SandboxError) as excinfo:
        await sandbox.run("raise RuntimeError('boom')")
    # THEN a SandboxError carries the traceback and a nonzero exit code
    assert excinfo.value.exit_code not in (None, 0)
    assert "boom" in excinfo.value.stderr


@pytest.mark.asyncio
async def test_timeout_raises_sandbox_timeout() -> None:
    # GIVEN a snippet that outlives the timeout
    sandbox = LocalProcessSandbox()
    # WHEN it runs with a short cap
    # THEN SandboxTimeout is raised (and the subprocess is torn down)
    with pytest.raises(SandboxTimeout):
        await sandbox.run("import time; time.sleep(60)", timeout_s=1.0)


@pytest.mark.asyncio
async def test_externals_not_supported() -> None:
    # GIVEN an externals mapping (CodeMode-style tool injection)
    sandbox = LocalProcessSandbox()
    # WHEN passed to run
    # THEN NotImplementedError mirrors the workload backend's contract
    with pytest.raises(NotImplementedError):
        await sandbox.run("_return = 1", externals={"tool": object()})


@pytest.mark.asyncio
async def test_non_json_return_value_is_stringified() -> None:
    # GIVEN a `_return` value that is not JSON-serializable
    sandbox = LocalProcessSandbox()
    # WHEN it runs
    result = await sandbox.run("import datetime\n_return = datetime.date(2026, 1, 2)")
    # THEN the value is stringified rather than crashing the run
    assert result.return_value == "2026-01-02"
