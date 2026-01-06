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

"""Tests for MCP elicitation capabilities announcement.

According to the MCP specification:
- Servers MUST announce capabilities they support
- Servers do NOT announce elicitation capability (it's a CLIENT capability)
- Clients MUST announce elicitation capability if they want to receive elicitation requests
- Elicitation supports form and url modes - clients announce which they support
- Servers MUST NOT send elicitation requests if client doesn't announce support
- Form mode: structured data with JSON schemas (for non-sensitive info)
- URL mode: redirects to external URLs (for sensitive info like credentials)

These tests verify strict compliance with the MCP specification.
Note: The MCP SDK doesn't currently support specifying form/url modes directly,
so we test that the server respects whether the client provides an elicitation callback.
"""

import pytest
from mcp.types import ElicitRequestParams
from mcp.types import ElicitResult

from datarobot_genai.drmcp import integration_test_mcp_session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "has_elicitation_callback,should_allow_elicitation",
    [
        # (client has callback, server can send elicitation)
        (False, False),  # No callback -> server MUST NOT send elicitation
        (True, True),  # Has callback -> server CAN send elicitation
    ],
)
async def test_elicitation_capability_negotiation(
    has_elicitation_callback,
    should_allow_elicitation,
):
    """
    Test elicitation capability negotiation between client and server.

    Verifies:
    1. Server MUST announce capabilities
    2. Server does NOT announce elicitation (it's a client capability)
    3. Server can send elicitation if client provides callback
    4. Server MUST NOT send elicitation if client doesn't provide callback
    """

    async def elicitation_handler(context, params: ElicitRequestParams) -> ElicitResult:
        """Decline all elicitation requests."""
        return ElicitResult(action="decline")

    callback = elicitation_handler if has_elicitation_callback else None

    async with integration_test_mcp_session(elicitation_callback=callback) as session:
        # Get the init result stored by integration_test_mcp_session
        init_result = session._init_result  # type: ignore[attr-defined]

        # Verify server MUST return capabilities
        assert init_result is not None, "Server MUST return initialize result"
        assert init_result.capabilities is not None, (
            "Server MUST return capabilities according to MCP specification"
        )

        # Verify server does NOT announce elicitation (it's a client capability)
        capabilities_dict = init_result.capabilities.model_dump(exclude_none=True)
        assert "elicitation" not in capabilities_dict, (
            f"Server should NOT announce elicitation capability. Got: {capabilities_dict}"
        )

        # Verify the test tool is available
        tools_result = await session.list_tools()
        tool_names = [tool.name for tool in tools_result.tools]
        assert "get_user_greeting" in tool_names, "Test tool should be available"

        # Call the tool without username - it will try to use elicitation
        result = await session.call_tool("get_user_greeting", {})

        if should_allow_elicitation:
            # Client supports elicitation - server CAN send elicitation request
            # Tool should not fail with capability error (may wait for user response)
            if result.isError:
                error_text = (
                    result.content[0].text
                    if result.content and hasattr(result.content[0], "text")
                    else str(result.content)
                )
                # Should not be a capability/elicitation support error
                assert not (
                    "capability" in error_text.lower()
                    and (
                        "not supported" in error_text.lower()
                        or "does not support" in error_text.lower()
                    )
                ), (
                    f"Tool should not fail with capability error when client supports "
                    f"elicitation. Got: {error_text}"
                )
        else:
            # Client doesn't support elicitation - server MUST NOT send elicitation
            # According to MCP spec, tool should return a non-error response (no-op)
            # indicating that elicitation is not supported
            assert not result.isError, (
                "Tool call should NOT fail when client doesn't support elicitation. "
                "According to MCP spec, it should return a no-op response."
            )
            result_text = (
                result.content[0].text
                if result.content and hasattr(result.content[0], "text")
                else str(result.content)
            )
            # Verify the response indicates elicitation is not supported
            assert "elicitation" in result_text.lower() or "skipped" in result_text.lower(), (
                f"Response should mention elicitation or be marked as skipped. Got: {result_text}"
            )
