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

"""Tests for client-side elicitation handling.

These tests verify that the client properly handles elicitation requests:
- Accepting elicitation with form data
- Declining elicitation requests
- Cancelling elicitation requests
- Handling different response types (string, int, bool, etc.)
"""

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import ElicitRequestParams
from mcp.types import ElicitResult

from datarobot_genai.drmcp import get_headers

# Import test tool to register it
from . import elicitation_test_tool  # noqa: F401


@pytest.mark.asyncio
async def test_client_accepts_elicitation(http_mcp_server):
    """Test that client can accept elicitation requests."""
    accepted_value = "testuser"

    async def elicitation_handler(context, params: ElicitRequestParams) -> ElicitResult:
        """Accept elicitation with test value."""
        return ElicitResult(action="accept", content={"value": accepted_value})

    async with streamablehttp_client(url=http_mcp_server, headers=get_headers()) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(
            read_stream, write_stream, elicitation_callback=elicitation_handler
        ) as session:
            await session.initialize()

            # Call tool that requires elicitation
            result = await session.call_tool("get_user_greeting", {})

            # Should succeed with accepted value
            assert not result.isError, (
                f"Tool should succeed with accepted elicitation. Got: {result.content}"
            )

            # Verify the result contains the accepted username
            result_text = (
                result.content[0].text
                if result.content and hasattr(result.content[0], "text")
                else str(result.content)
            )
            assert accepted_value in result_text.lower(), (
                f"Result should contain accepted username '{accepted_value}'. Got: {result_text}"
            )


@pytest.mark.asyncio
async def test_client_declines_elicitation(http_mcp_server):
    """Test that client can decline elicitation requests."""

    async def elicitation_handler(context, params: ElicitRequestParams) -> ElicitResult:
        """Decline elicitation."""
        return ElicitResult(action="decline")

    async with streamablehttp_client(url=http_mcp_server, headers=get_headers()) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(
            read_stream, write_stream, elicitation_callback=elicitation_handler
        ) as session:
            await session.initialize()

            # Call tool that requires elicitation
            result = await session.call_tool("get_user_greeting", {})

            # Should return error indicating decline
            assert result.isError or (
                not result.isError and "declined" in str(result.content).lower()
            ), f"Tool should handle declined elicitation. Got: {result.content}"


@pytest.mark.asyncio
async def test_client_cancels_elicitation(http_mcp_server):
    """Test that client can cancel elicitation requests."""

    async def elicitation_handler(context, params: ElicitRequestParams) -> ElicitResult:
        """Cancel elicitation."""
        return ElicitResult(action="cancel")

    async with streamablehttp_client(url=http_mcp_server, headers=get_headers()) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(
            read_stream, write_stream, elicitation_callback=elicitation_handler
        ) as session:
            await session.initialize()

            # Call tool that requires elicitation
            result = await session.call_tool("get_user_greeting", {})

            # Should return error indicating cancellation
            assert result.isError or (
                not result.isError
                and (
                    "cancel" in str(result.content).lower()
                    or "cancelled" in str(result.content).lower()
                )
            ), f"Tool should handle cancelled elicitation. Got: {result.content}"


@pytest.mark.asyncio
async def test_client_handles_form_schema(http_mcp_server):
    """Test that client can handle elicitation requests with JSON schemas."""

    async def elicitation_handler(context, params: ElicitRequestParams) -> ElicitResult:
        """Validate and return form data."""
        # Verify schema is provided
        assert params.requestedSchema is not None, "Schema should be provided"

        # Return form data matching schema
        return ElicitResult(
            action="accept",
            content={"value": "testuser"},  # Simple form data
        )

    async with streamablehttp_client(url=http_mcp_server, headers=get_headers()) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(
            read_stream, write_stream, elicitation_callback=elicitation_handler
        ) as session:
            await session.initialize()

            # Call tool that requires elicitation
            result = await session.call_tool("get_user_greeting", {})

            # Should succeed
            assert not result.isError, (
                f"Tool should succeed with form schema. Got: {result.content}"
            )
