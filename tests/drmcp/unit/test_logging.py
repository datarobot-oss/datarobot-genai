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

import logging

import pytest
from fastmcp.exceptions import ToolError as FastMCPToolError
from fastmcp.exceptions import ValidationError as FastMCPValidationError
from pydantic import TypeAdapter

from datarobot_genai.drmcp.core.logging import SecretRedactingFormatter
from datarobot_genai.drmcp.core.logging import log_execution
from datarobot_genai.drmcputils.exceptions import ToolError as DRToolError
from datarobot_genai.drmcputils.exceptions import ToolErrorKind


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


_MESSAGES_CONTAINING_SECRET = [
    "sk-a1b1c1111111111A1B1C1111119111191111111181111117",  # OpenAI key
    "sk-proj-a1b1c111_111111A1B1C111111911119111-11_181111117991122aaa",  # newer OpenAI key
    "AKIAA1B11C11111D1111",  # AWS key
    "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.c2lnbmF0dXJl",  # JWT
    "Bearer abcDEF123456789xyz",  # Authorization header value
    "basic dXNlcjpwYXNzd29yZA==",  # Basic auth value (case-insensitive)
    "password=Sup3rS3cret!",  # key=value assignment
    "api_key: 12345secretvalue",  # key: value assignment
    "DATAROBOT_API_TOKEN=NjM4someToken-value_here",  # env-style token assignment
]

# Regression guard for the removed catch-all ``([a-zA-Z0-9]{20,})``: operational
# identifiers must survive redaction so logs stay debuggable.
_MESSAGES_WITHOUT_SECRETS = [
    "UserMCPProvider: API client not yet initialised",  # class name
    "request_id=abc123def456ghi789jkl012 completed",  # long request id
    "deployment 6513f86cd799439011abcdef failed",  # 24-hex ObjectId
    "trace_id=4bf92f3577b34da6a3ce929d0e0e4736",  # trace id
    "tokens=1500 completion_tokens=300",  # LLM usage counters (plural 'tokens')
    "Registered tool datarobot_docs_search_documentation",  # long tool name
]


def _format_with_redaction(msg: str) -> str:
    formatter = SecretRedactingFormatter("%(message)s")
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg=msg, args=None, exc_info=None
    )
    return formatter.format(record)


@pytest.mark.parametrize("msg", _MESSAGES_CONTAINING_SECRET)
def test_secret_redacting_formatter_redacts_secrets(msg: str) -> None:
    # GIVEN a log message containing a secret
    # WHEN it is formatted / THEN the secret is fully replaced
    assert _format_with_redaction(msg) == "[REDACTED]"


@pytest.mark.parametrize("msg", _MESSAGES_WITHOUT_SECRETS)
def test_secret_redacting_formatter_preserves_operational_identifiers(msg: str) -> None:
    # GIVEN a log line with ids/class names but no secret
    # WHEN it is formatted / THEN the message is untouched
    assert _format_with_redaction(msg) == msg


def test_secret_redacting_formatter_redacts_secret_within_context() -> None:
    # GIVEN a realistic line mixing a secret with operational context
    msg = "Auth failed for deployment 6513f86cd799439011abcdef: Authorization: Bearer eyJa.eyJb.sig"

    redacted = _format_with_redaction(msg)

    # THEN the token is gone but the surrounding context survives
    assert "eyJa.eyJb.sig" not in redacted
    assert "[REDACTED]" in redacted
    assert "6513f86cd799439011abcdef" in redacted
