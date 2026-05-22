# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `datarobot_genai.drtools.core.mode`."""

from unittest.mock import patch

import pytest

from datarobot_genai.drtools.core.mode import MCP_MODE_HEADER
from datarobot_genai.drtools.core.mode import MCPMode
from datarobot_genai.drtools.core.mode import get_mcp_mode


class TestMCPMode:
    def test_header_value(self) -> None:
        assert MCP_MODE_HEADER == "x-datarobot-mcp-mode"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("tools", MCPMode.TOOLS),
            ("code_execute", MCPMode.CODE_EXECUTE),
            ("TOOLS", MCPMode.TOOLS),
            ("Code_Execute", MCPMode.CODE_EXECUTE),
            ("  tools  ", MCPMode.TOOLS),
            ("", MCPMode.TOOLS),
            ("nonsense", MCPMode.TOOLS),
            ("code-execute", MCPMode.TOOLS),  # hyphen mismatch falls back
        ],
    )
    def test_from_header(self, raw: str, expected: MCPMode) -> None:
        assert MCPMode.from_header(raw) is expected

    @pytest.mark.parametrize(
        "mode,expected",
        [
            (MCPMode.TOOLS, "tools"),
            (MCPMode.CODE_EXECUTE, "code_execute"),
        ],
    )
    def test_to_header(self, mode: MCPMode, expected: str) -> None:
        assert mode.to_header() == expected

    def test_roundtrip(self) -> None:
        for mode in MCPMode:
            assert MCPMode.from_header(mode.to_header()) is mode


class TestGetMcpMode:
    """`get_mcp_mode()` reads via FastMCP's `get_http_headers`."""

    def test_defaults_to_tools_when_no_headers(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.mode._get_http_headers",
            return_value={},
        ):
            assert get_mcp_mode() is MCPMode.TOOLS

    def test_returns_tools_when_header_says_tools(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.mode._get_http_headers",
            return_value={MCP_MODE_HEADER: "tools"},
        ):
            assert get_mcp_mode() is MCPMode.TOOLS

    def test_returns_code_execute_when_header_set(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.mode._get_http_headers",
            return_value={MCP_MODE_HEADER: "code_execute"},
        ):
            assert get_mcp_mode() is MCPMode.CODE_EXECUTE

    def test_unknown_header_value_falls_back_to_tools(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.mode._get_http_headers",
            return_value={MCP_MODE_HEADER: "gibberish"},
        ):
            assert get_mcp_mode() is MCPMode.TOOLS

    def test_other_headers_ignored(self) -> None:
        with patch(
            "datarobot_genai.drtools.core.mode._get_http_headers",
            return_value={
                "x-other-header": "code_execute",
                "authorization": "Bearer xyz",
            },
        ):
            assert get_mcp_mode() is MCPMode.TOOLS
