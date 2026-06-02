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

"""Tests for DataRobot tools client (get_datarobot_access_token)."""

from unittest.mock import patch

import pytest

from datarobot_genai.drtools.core.clients.datarobot import get_datarobot_access_token
from datarobot_genai.drtools.core.exceptions import ToolError


class TestGetDatarobotAccessToken:
    """Test get_datarobot_access_token function."""

    def test_returns_token_when_resolve_token_from_headers_returns_token(self) -> None:
        """Test successful token retrieval from headers."""
        with patch(
            "datarobot_genai.drtools.core.clients.datarobot.resolve_token_from_headers",
            return_value="bearer-token-123",
        ):
            result = get_datarobot_access_token()
        assert result == "bearer-token-123"
        assert isinstance(result, str)

    def test_raises_tool_error_when_no_token(self) -> None:
        """Test that ToolError is raised when resolve_token_from_headers returns None."""
        with patch(
            "datarobot_genai.drtools.core.clients.datarobot.resolve_token_from_headers",
            return_value=None,
        ):
            with pytest.raises(ToolError) as exc_info:
                get_datarobot_access_token()
        assert "DataRobot API token not found" in str(exc_info.value)
        assert "Authorization" in str(exc_info.value) or "x-datarobot-api-token" in str(
            exc_info.value
        )

    def test_raises_tool_error_when_empty_token(self) -> None:
        """Test that ToolError is raised when resolve_token_from_headers returns empty string."""
        with patch(
            "datarobot_genai.drtools.core.clients.datarobot.resolve_token_from_headers",
            return_value="",
        ):
            with pytest.raises(ToolError) as exc_info:
                get_datarobot_access_token()
        assert "DataRobot API token not found" in str(exc_info.value)
