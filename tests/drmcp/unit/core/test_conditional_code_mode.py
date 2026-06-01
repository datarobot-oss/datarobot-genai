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

"""Tests for ``ConditionalCodeMode`` and ``MCPMode`` — per-request CodeMode gating."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp import FastMCP
from fastmcp.experimental.transforms.code_mode import _ensure_async

from datarobot_genai.drmcp.core.conditional_code_mode import ConditionalCodeMode
from datarobot_genai.drmcp.core.conditional_code_mode import MCPMode
from datarobot_genai.drmcp.core.conditional_code_mode import initialize_code_mode

MODULE = "datarobot_genai.drmcp.core.conditional_code_mode"


class TestMCPMode:
    @pytest.fixture
    def code_mode_header_key(self) -> str:
        return "x-datarobot-mcp-mode"

    @pytest.fixture
    def mock_get_fast_mcp_headers(self) -> Iterator[Mock]:
        with patch(f"{MODULE}.get_fast_mcp_http_headers") as mock_func:
            mock_func.return_value = {}
            yield mock_func

    def test_defaults_to_tools_when_no_headers(self, mock_get_fast_mcp_headers: Mock) -> None:
        mock_get_fast_mcp_headers.return_value = {}

        assert MCPMode.from_current_http_request_headers() == MCPMode.TOOLS

    @pytest.mark.parametrize("tools", ["TOOLS", "tools"])
    def test_returns_tools_when_header_says_tools(
        self, tools: str, mock_get_fast_mcp_headers: Mock, code_mode_header_key: str
    ) -> None:
        mock_get_fast_mcp_headers.return_value = {code_mode_header_key: tools}

        assert MCPMode.from_current_http_request_headers() == MCPMode.TOOLS

    @pytest.mark.parametrize("code_execute", ["code_execute", "CODE_EXECUTE"])
    def test_returns_code_execute_when_header_set(
        self, code_execute: str, mock_get_fast_mcp_headers: Mock, code_mode_header_key: str
    ) -> None:
        mock_get_fast_mcp_headers.return_value = {code_mode_header_key: code_execute}

        assert MCPMode.from_current_http_request_headers() == MCPMode.CODE_EXECUTE

    def test_unknown_header_value_falls_back_to_tools(
        self, mock_get_fast_mcp_headers: Mock, code_mode_header_key: str
    ) -> None:
        mock_get_fast_mcp_headers.return_value = {code_mode_header_key: "ooops"}

        assert MCPMode.from_current_http_request_headers() == MCPMode.TOOLS

    def test_other_headers_ignored(
        self, mock_get_fast_mcp_headers: Mock, code_mode_header_key: str
    ) -> None:
        mock_get_fast_mcp_headers.return_value = {
            code_mode_header_key + "x": MCPMode.CODE_EXECUTE.name
        }

        assert MCPMode.from_current_http_request_headers() == MCPMode.TOOLS


class TestInitializeCodeMode:
    def test_registers_transform(self) -> None:
        mcp = Mock()
        initialize_code_mode(mcp)
        mcp.add_transform.assert_called_once()
        assert isinstance(mcp.add_transform.call_args[0][0], ConditionalCodeMode)


class _UnsafeTestSandboxProvider:
    async def run(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        external_functions: dict[str, Any] | None = None,
    ) -> Any:
        namespace: dict[str, Any] = {}
        if inputs:
            namespace.update(inputs)
        if external_functions:
            namespace.update({k: _ensure_async(v) for k, v in external_functions.items()})
        wrapped = "async def __test_main__():\n"
        for line in code.splitlines():
            wrapped += f"    {line}\n" if line.strip() else "    pass\n"
        exec(wrapped, namespace)  # noqa: S102 — test-only
        return await namespace["__test_main__"]()


class TestConditionalCodeMode:
    @staticmethod
    def make_server() -> FastMCP:
        mcp = FastMCP("ConditionalCodeMode test")

        @mcp.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        @mcp.tool
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        mcp.add_transform(ConditionalCodeMode(sandbox_provider=_UnsafeTestSandboxProvider()))
        return mcp

    @pytest.fixture
    def mock_mode(self) -> Iterator[Mock]:
        with patch.object(MCPMode, "from_current_http_request_headers") as m:
            m.return_value = MCPMode.TOOLS
            yield m

    @pytest.mark.asyncio
    async def test_tools_mode_exposes_real_catalog(self, mock_mode: Mock) -> None:
        mock_mode.return_value = MCPMode.TOOLS
        mcp = self.make_server()

        names = {t.name for t in await mcp.list_tools(run_middleware=False)}

        assert {"add", "greet"} <= names
        assert "execute" not in names
        assert "search" not in names
        assert "get_schema" not in names

    @pytest.mark.asyncio
    async def test_code_execute_mode_collapses_catalog(self, mock_mode: Mock) -> None:
        mock_mode.return_value = MCPMode.CODE_EXECUTE
        mcp = self.make_server()

        names = {t.name for t in await mcp.list_tools(run_middleware=False)}

        assert names == {"search", "get_schema", "execute"}

    @pytest.mark.asyncio
    async def test_mode_switch_between_calls(self, mock_mode: Mock) -> None:
        mcp = self.make_server()

        mock_mode.return_value = MCPMode.TOOLS
        tools_view = {t.name for t in await mcp.list_tools(run_middleware=False)}
        assert {"add", "greet"} <= tools_view

        mock_mode.return_value = MCPMode.CODE_EXECUTE
        code_view = {t.name for t in await mcp.list_tools(run_middleware=False)}
        assert code_view == {"search", "get_schema", "execute"}

        mock_mode.return_value = MCPMode.TOOLS
        back_view = {t.name for t in await mcp.list_tools(run_middleware=False)}
        assert back_view == tools_view

    @pytest.mark.asyncio
    async def test_get_tool_passes_through_in_tools_mode(self, mock_mode: Mock) -> None:
        mock_mode.return_value = MCPMode.TOOLS
        mcp = self.make_server()

        add_tool = await mcp.get_tool("add")
        assert add_tool is not None
        assert add_tool.name == "add"

        assert await mcp.get_tool("execute") is None
        assert await mcp.get_tool("search") is None

    @pytest.mark.asyncio
    async def test_get_tool_resolves_meta_tools_in_code_execute_mode(self, mock_mode: Mock) -> None:
        mock_mode.return_value = MCPMode.CODE_EXECUTE
        mcp = self.make_server()

        for name in ("search", "get_schema", "execute"):
            tool = await mcp.get_tool(name)
            assert tool is not None, f"Expected {name} to be resolvable in code_execute mode"
            assert tool.name == name

    @pytest.mark.asyncio
    async def test_no_header_defaults_to_tools_mode(self) -> None:
        mcp = self.make_server()

        # Don't patch get_mcp_mode — let the real implementation run with no HTTP ctx.
        # _get_http_headers returns {} outside a request, so get_mcp_mode → TOOLS.
        listed = await mcp.list_tools(run_middleware=False)
        names = {t.name for t in listed}
        assert {"add", "greet"} <= names
        assert "execute" not in names
