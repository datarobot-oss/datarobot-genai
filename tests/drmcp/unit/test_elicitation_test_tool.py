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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastmcp import Context
from fastmcp.server.context import AcceptedElicitation
from fastmcp.server.context import CancelledElicitation
from fastmcp.server.context import DeclinedElicitation
from mcp.types import ClientCapabilities
from mcp.types import ElicitationCapability

from tests.drmcp.integration.elicitation_test_tool import get_user_greeting


class TestGetUserGreeting:
    """Test cases for get_user_greeting tool."""

    @pytest.mark.asyncio
    async def test_get_user_greeting_with_username_provided(self) -> None:
        """Test get_user_greeting when username is provided (no elicitation needed)."""
        mock_ctx = MagicMock(spec=Context)

        result = await get_user_greeting(mock_ctx, username="testuser")

        assert result["status"] == "success"
        assert result["username"] == "testuser"
        assert "testuser" in result["message"]
        # Should not check capabilities or use elicitation
        assert (
            not hasattr(mock_ctx, "session") or not mock_ctx.session.check_client_capability.called
        )

    @pytest.mark.asyncio
    async def test_get_user_greeting_without_username_no_elicitation_support(
        self,
    ) -> None:
        """Test get_user_greeting when username not provided and client doesn't support elicitation."""  # noqa: E501
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.session.check_client_capability.side_effect = AttributeError("No method")

        result = await get_user_greeting(mock_ctx, username=None)

        assert result["status"] == "skipped"
        assert result["elicitation_supported"] is False
        assert "not supported" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_get_user_greeting_without_username_check_capability_typeerror(
        self,
    ) -> None:
        """Test get_user_greeting when check_client_capability raises TypeError."""
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.session.check_client_capability.side_effect = TypeError("Invalid type")

        result = await get_user_greeting(mock_ctx, username=None)

        assert result["status"] == "skipped"
        assert result["elicitation_supported"] is False

    @pytest.mark.asyncio
    async def test_get_user_greeting_without_username_check_capability_returns_false(
        self,
    ) -> None:
        """Test get_user_greeting when check_client_capability returns False."""
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.session.check_client_capability.return_value = False

        result = await get_user_greeting(mock_ctx, username=None)

        assert result["status"] == "skipped"
        assert result["elicitation_supported"] is False

    @pytest.mark.asyncio
    async def test_get_user_greeting_with_elicitation_accepted(self) -> None:
        """Test get_user_greeting when elicitation is accepted."""
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.session.check_client_capability.return_value = True
        mock_ctx.elicit = AsyncMock(return_value=AcceptedElicitation(data="accepted_user"))

        result = await get_user_greeting(mock_ctx, username=None)

        assert result["status"] == "success"
        assert result["username"] == "accepted_user"
        assert "accepted_user" in result["message"]
        mock_ctx.elicit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_greeting_with_elicitation_declined(self) -> None:
        """Test get_user_greeting when elicitation is declined."""
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.session.check_client_capability.return_value = True
        mock_ctx.elicit = AsyncMock(return_value=DeclinedElicitation())

        result = await get_user_greeting(mock_ctx, username=None)

        assert result["status"] == "error"
        assert result["error"] == "Username declined by user"
        assert "cannot generate greeting" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_get_user_greeting_with_elicitation_cancelled(self) -> None:
        """Test get_user_greeting when elicitation is cancelled."""
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.session.check_client_capability.return_value = True
        mock_ctx.elicit = AsyncMock(return_value=CancelledElicitation())

        result = await get_user_greeting(mock_ctx, username=None)

        assert result["status"] == "error"
        assert result["error"] == "Operation cancelled"
        assert "cancelled" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_get_user_greeting_with_elicitation_no_session_attribute(self) -> None:
        """Test get_user_greeting when ctx has no session attribute."""
        mock_ctx = MagicMock(spec=Context)
        del mock_ctx.session

        result = await get_user_greeting(mock_ctx, username=None)

        assert result["status"] == "skipped"
        assert result["elicitation_supported"] is False

    @pytest.mark.asyncio
    async def test_get_user_greeting_verifies_capability_check(self) -> None:
        """Test that get_user_greeting properly checks client capability."""
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.session.check_client_capability.return_value = True
        mock_ctx.elicit = AsyncMock(return_value=AcceptedElicitation(data="user123"))

        await get_user_greeting(mock_ctx, username=None)

        # Verify check_client_capability was called with correct arguments
        mock_ctx.session.check_client_capability.assert_called_once()
        call_args = mock_ctx.session.check_client_capability.call_args[0][0]
        assert isinstance(call_args, ClientCapabilities)
        assert call_args.elicitation is not None
        assert isinstance(call_args.elicitation, ElicitationCapability)
