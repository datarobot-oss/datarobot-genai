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

import pytest
from fastmcp.exceptions import ToolError as FastMCPToolError
from fastmcp.exceptions import ValidationError as FastMCPValidationError
from pydantic import TypeAdapter

from datarobot_genai.drmcp.core.logging import log_execution
from datarobot_genai.drtools.core.exceptions import ToolError as DRToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind


@pytest.mark.asyncio
async def test_log_execution_wraps_dr_tool_error_with_kind_prefix() -> None:
    @log_execution
    async def failing_tool() -> None:
        raise DRToolError("bad input", kind=ToolErrorKind.VALIDATION)

    with pytest.raises(FastMCPToolError) as exc_info:
        await failing_tool()
    assert str(exc_info.value) == "[validation] bad input"


@pytest.mark.asyncio
async def test_log_execution_passes_through_fastmcp_tool_error() -> None:
    @log_execution
    async def raises_fm() -> None:
        raise FastMCPToolError("already formatted")

    with pytest.raises(FastMCPToolError) as exc_info:
        await raises_fm()
    assert str(exc_info.value) == "already formatted"


@pytest.mark.asyncio
async def test_log_execution_wraps_pydantic_validation_as_schema() -> None:
    @log_execution
    async def pydantic_fail() -> None:
        TypeAdapter(int).validate_python("not-an-int")

    with pytest.raises(FastMCPToolError) as exc_info:
        await pydantic_fail()
    assert str(exc_info.value).startswith("[schema]")


@pytest.mark.asyncio
async def test_log_execution_wraps_fastmcp_validation_as_schema() -> None:
    @log_execution
    async def fm_validation_fail() -> None:
        raise FastMCPValidationError("invalid tool parameters")

    with pytest.raises(FastMCPToolError) as exc_info:
        await fm_validation_fail()
    assert str(exc_info.value).startswith("[schema]")


@pytest.mark.asyncio
async def test_log_execution_wraps_404_like_exception_as_not_found() -> None:
    @log_execution
    async def raises_404_style() -> None:
        raise Exception("404 client error: {'message': 'Not Found'}")

    with pytest.raises(FastMCPToolError) as exc_info:
        await raises_404_style()
    assert str(exc_info.value).startswith("[not_found]")


@pytest.mark.asyncio
async def test_log_execution_wraps_generic_exception_as_internal() -> None:
    @log_execution
    async def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(FastMCPToolError) as exc_info:
        await boom()
    assert "[internal]" in str(exc_info.value)
    assert "boom" in str(exc_info.value)


@pytest.mark.asyncio
async def test_log_execution_wraps_exception_with_status_code_as_upstream() -> None:
    class HttpishError(Exception):
        status_code = 502

        def __str__(self) -> str:
            return "bad gateway"

    @log_execution
    async def upstream() -> None:
        raise HttpishError()

    with pytest.raises(FastMCPToolError) as exc_info:
        await upstream()
    assert str(exc_info.value).startswith("[upstream]")


@pytest.mark.asyncio
async def test_log_execution_wraps_exception_with_403_as_authentication() -> None:
    class ForbiddenError(Exception):
        status_code = 403

        def __str__(self) -> str:
            return "forbidden"

    @log_execution
    async def forbidden() -> None:
        raise ForbiddenError()

    with pytest.raises(FastMCPToolError) as exc_info:
        await forbidden()
    assert str(exc_info.value).startswith("[authentication]")
