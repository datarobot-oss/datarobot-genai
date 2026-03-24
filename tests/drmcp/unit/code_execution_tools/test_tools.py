# Copyright 2025 DataRobot, Inc.
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

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from datarobot_genai.drtools.code_execution.tools import InProcessSandbox
from datarobot_genai.drtools.code_execution.tools import NoopSandbox
from datarobot_genai.drtools.code_execution.tools import execute_code
from datarobot_genai.drtools.code_execution.tools import set_sandbox_provider


@pytest.mark.asyncio
async def test_noop_sandbox_returns_error() -> None:
    sandbox = NoopSandbox()
    result = await sandbox.execute("print('hi')", "session1")
    assert result["error"] is not None
    assert "not available" in result["error"]
    assert result["stdout"] == ""
    assert result["result"] is None


@pytest.mark.asyncio
async def test_in_process_sandbox_simple_code() -> None:
    sandbox = InProcessSandbox()
    result = await sandbox.execute("print('hello')", "session1")
    assert result["error"] is None
    assert "hello" in result["stdout"]


@pytest.mark.asyncio
async def test_in_process_sandbox_result_variable() -> None:
    sandbox = InProcessSandbox()
    result = await sandbox.execute("result = 42", "session1")
    assert result["error"] is None
    assert result["result"] == 42


@pytest.mark.asyncio
async def test_in_process_sandbox_syntax_error() -> None:
    sandbox = InProcessSandbox()
    result = await sandbox.execute("def bad(:", "session1")
    assert result["error"] is not None
    assert "SyntaxError" in result["error"]


@pytest.mark.asyncio
async def test_in_process_sandbox_runtime_error() -> None:
    sandbox = InProcessSandbox()
    result = await sandbox.execute("raise ValueError('oops')", "session1")
    assert result["error"] is not None
    assert "ValueError" in result["error"]


@pytest.mark.asyncio
async def test_in_process_sandbox_timeout() -> None:
    sandbox = InProcessSandbox()
    result = await sandbox.execute("import time; time.sleep(10)", "session1", timeout_seconds=1)
    assert result["error"] is not None
    assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_execute_code_with_noop_sandbox() -> None:
    set_sandbox_provider(NoopSandbox())
    result = await execute_code(code="print('test')")
    assert isinstance(result, ToolResult)
    assert "not available" in result.structured_content["error"]


@pytest.mark.asyncio
async def test_execute_code_with_in_process_sandbox() -> None:
    set_sandbox_provider(InProcessSandbox())
    result = await execute_code(code="result = 'ok'")
    assert isinstance(result, ToolResult)
    assert result.structured_content["result"] == "ok"
    assert result.structured_content["error"] is None
    # Reset to noop for safety
    set_sandbox_provider(NoopSandbox())


@pytest.mark.asyncio
async def test_execute_code_missing_code() -> None:
    with pytest.raises(ToolError, match="Code must be provided"):
        await execute_code()
