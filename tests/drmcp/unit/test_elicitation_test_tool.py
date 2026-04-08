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

"""Unit tests for elicitation_test_tool.py module."""

from contextlib import contextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastmcp import Context
from fastmcp.server.context import AcceptedElicitation
from fastmcp.server.context import CancelledElicitation
from fastmcp.server.context import DeclinedElicitation
from fastmcp.server.context import _current_context
from fastmcp.tools.function_tool import FunctionTool

from datarobot_genai.drmcp.core.mcp_instance import mcp

# Import the tool module to register the tool
from datarobot_genai.drmcp.test_utils import elicitation_test_tool  # noqa: F401


@contextmanager
def _mock_context(**kwargs):
    """Set a mock Context in fastmcp's context variable for tool.run() dependency injection."""
    mock_ctx = MagicMock(spec=Context, **kwargs)
    token = _current_context.set(mock_ctx)
    try:
        yield mock_ctx
    finally:
        _current_context.reset(token)


class TestGetUserGreeting:
    """Test cases for get_user_greeting tool."""

    @pytest.mark.asyncio
    async def test_get_user_greeting_with_username_provided(self) -> None:
        """Test get_user_greeting when username is provided (no elicitation needed)."""
        # Get the tool from the MCP instance
        tool = await mcp.get_tool("get_user_greeting")
        assert isinstance(tool, FunctionTool)

        with _mock_context():
            tool_result = await tool.run({"username": "testuser"})
            result = tool_result.structured_content

        assert result["status"] == "success"
        assert result["username"] == "testuser"
        assert "testuser" in result["message"]

    @pytest.mark.asyncio
    async def test_get_user_greeting_with_elicitation_accepted(self) -> None:
        """Test get_user_greeting when elicitation is accepted."""
        tool = await mcp.get_tool("get_user_greeting")
        assert isinstance(tool, FunctionTool)

        with _mock_context() as mock_ctx:
            mock_ctx.elicit = AsyncMock(return_value=AcceptedElicitation(data="accepted_user"))
            tool_result = await tool.run({})
            result = tool_result.structured_content

        assert result["status"] == "success"
        assert result["username"] == "accepted_user"
        assert "accepted_user" in result["message"]
        mock_ctx.elicit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_greeting_with_elicitation_declined(self) -> None:
        """Test get_user_greeting when elicitation is declined."""
        tool = await mcp.get_tool("get_user_greeting")
        assert isinstance(tool, FunctionTool)

        with _mock_context() as mock_ctx:
            mock_ctx.elicit = AsyncMock(return_value=DeclinedElicitation())
            tool_result = await tool.run({})
            result = tool_result.structured_content

        assert result["status"] == "error"
        assert result["error"] == "Username declined by user"
        assert "cannot generate greeting" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_get_user_greeting_with_elicitation_cancelled(self) -> None:
        """Test get_user_greeting when elicitation is cancelled."""
        tool = await mcp.get_tool("get_user_greeting")
        assert isinstance(tool, FunctionTool)

        with _mock_context() as mock_ctx:
            mock_ctx.elicit = AsyncMock(return_value=CancelledElicitation())
            tool_result = await tool.run({})
            result = tool_result.structured_content

        assert result["status"] == "error"
        assert result["error"] == "Operation cancelled"
        assert "cancelled" in result["message"].lower()
