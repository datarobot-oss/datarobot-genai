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

# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License == distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

import pytest

from datarobot_genai.drtools.core.mode import MCPMode


@pytest.fixture()
def module_under_test():
    return "datarobot_genai.drtools.core.mode"


class TestGetMcpMode:
    @pytest.fixture
    def code_mode_header_key(self):
        return "x-datarobot-mcp-mode"

    @pytest.fixture
    def mock_get_fast_mcp_headers(self, module_under_test):
        with patch(f"{module_under_test}.get_fast_mcp_http_headers") as mock_func:
            mock_func.return_value = {}
            yield mock_func

    def test_defaults_to_tools_when_no_headers(self, mock_get_fast_mcp_headers) -> None:
        mock_get_fast_mcp_headers.return_value = {}

        actual = MCPMode.from_current_http_request_headers()

        assert actual == MCPMode.TOOLS

    @pytest.mark.parametrize("tools", ["TOOLS", "tools"])
    def test_returns_tools_when_header_says_tools(
        self, tools, mock_get_fast_mcp_headers, code_mode_header_key
    ) -> None:
        mock_get_fast_mcp_headers.return_value = {code_mode_header_key: tools}

        actual = MCPMode.from_current_http_request_headers()

        assert actual == MCPMode.TOOLS

    @pytest.mark.parametrize("code_execute", ["code_execute", "CODE_EXECUTE"])
    def test_returns_code_execute_when_header_set(
        self, code_execute, mock_get_fast_mcp_headers, code_mode_header_key
    ) -> None:
        mock_get_fast_mcp_headers.return_value = {code_mode_header_key: code_execute}

        actual = MCPMode.from_current_http_request_headers()

        assert actual == MCPMode.CODE_EXECUTE

    def test_unknown_header_value_falls_back_to_tools(
        self, mock_get_fast_mcp_headers, code_mode_header_key
    ) -> None:
        mock_get_fast_mcp_headers.return_value = {code_mode_header_key: "ooops"}

        actual = MCPMode.from_current_http_request_headers()

        assert actual == MCPMode.TOOLS

    def test_other_headers_ignored(self, mock_get_fast_mcp_headers, code_mode_header_key) -> None:
        mock_get_fast_mcp_headers.return_value = {
            code_mode_header_key + "x": MCPMode.CODE_EXECUTE.name
        }

        actual = MCPMode.from_current_http_request_headers()

        assert actual == MCPMode.TOOLS
