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

"""Tests for per-request MCP catalog transforms (CodeMode + tool allowlist)."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp import FastMCP
from fastmcp.experimental.transforms.code_mode import _ensure_async

from datarobot_genai.drmcpbase.fastmcp_transforms import register_mcp_catalog_transform
from datarobot_genai.drmcpbase.fastmcp_transforms.transform import DataRobotMCPCatalogTransform
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCPRequestContext
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCPRequestMode
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import _request_context_cache
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import filter_tools_by_allowlist
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import get_header_case_insensitive
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import is_tool_name_allowed
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import parse_tool_allowlist_header


@pytest.fixture(autouse=True)
def _reset_request_context_cache() -> Iterator[None]:
    _request_context_cache.set(None)
    yield
    _request_context_cache.set(None)


@pytest.fixture
def utils_module() -> str:
    return "datarobot_genai.drmcpbase.fastmcp_transforms.utils"


@pytest.fixture
def transform_module() -> str:
    return "datarobot_genai.drmcpbase.fastmcp_transforms.transform"


class TestHeaderHelpers:
    def test_get_header_value_direct_lowercase_lookup(self) -> None:
        headers = {"x-datarobot-mcp-tools": "add,greet"}
        assert get_header_case_insensitive(headers, "x-datarobot-mcp-tools") == "add,greet"

    def test_get_header_value_fallback_for_mixed_case_keys(self) -> None:
        headers = {"X-DataRobot-MCP-Tools": "add,greet"}
        assert get_header_case_insensitive(headers, "x-datarobot-mcp-tools") == "add,greet"

    def test_get_header_value_missing(self) -> None:
        assert get_header_case_insensitive({}, "x-datarobot-mcp-tools") is None


class TestParseToolAllowlistHeader:
    def test_none_when_raw_is_none(self) -> None:
        assert parse_tool_allowlist_header(None) is None

    def test_parses_comma_separated_names(self) -> None:
        assert parse_tool_allowlist_header(" add , greet ") == frozenset({"add", "greet"})

    def test_none_when_empty_string(self) -> None:
        assert parse_tool_allowlist_header("   ") is None

    def test_none_when_only_commas(self) -> None:
        assert parse_tool_allowlist_header(",,,") is None


class TestMCPRequestMode:
    @pytest.fixture
    def mock_get_fast_mcp_headers(self, utils_module: str) -> Iterator[Mock]:
        with patch(f"{utils_module}.get_fast_mcp_http_headers") as mock_func:
            mock_func.return_value = {}
            yield mock_func

    def test_defaults_to_tools_when_no_headers(self, mock_get_fast_mcp_headers: Mock) -> None:
        mock_get_fast_mcp_headers.return_value = {}
        ctx = MCPRequestContext.from_current_http_request()
        assert ctx.mode == MCPRequestMode.TOOLS
        assert ctx.tool_allowlist is None

    @pytest.mark.parametrize("tools", ["TOOLS", "tools"])
    def test_returns_tools_when_header_says_tools(
        self, tools: str, mock_get_fast_mcp_headers: Mock
    ) -> None:
        mock_get_fast_mcp_headers.return_value = {"x-datarobot-mcp-mode": tools}
        assert MCPRequestContext.from_current_http_request().mode == MCPRequestMode.TOOLS

    @pytest.mark.parametrize("code_execute", ["code_execute", "CODE_EXECUTE"])
    def test_returns_code_execute_when_header_set(
        self, code_execute: str, mock_get_fast_mcp_headers: Mock
    ) -> None:
        mock_get_fast_mcp_headers.return_value = {"x-datarobot-mcp-mode": code_execute}
        assert MCPRequestContext.from_current_http_request().mode == MCPRequestMode.CODE_EXECUTE

    def test_unknown_header_value_falls_back_to_tools(
        self, mock_get_fast_mcp_headers: Mock
    ) -> None:
        mock_get_fast_mcp_headers.return_value = {"x-datarobot-mcp-mode": "ooops"}
        assert MCPRequestContext.from_current_http_request().mode == MCPRequestMode.TOOLS


class TestRegisterMcpCatalogTransform:
    def test_registers_transform(self) -> None:
        mcp = Mock()
        register_mcp_catalog_transform(mcp)
        mcp.add_transform.assert_called_once()
        assert isinstance(mcp.add_transform.call_args[0][0], DataRobotMCPCatalogTransform)


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


class TestDataRobotMCPCatalogTransform:
    @staticmethod
    def make_server() -> FastMCP:
        mcp = FastMCP("DataRobotMCPCatalogTransform test")

        @mcp.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        @mcp.tool
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        mcp.add_transform(
            DataRobotMCPCatalogTransform(sandbox_provider=_UnsafeTestSandboxProvider())
        )
        return mcp

    @pytest.fixture
    def mock_context(self, transform_module: str) -> Iterator[Mock]:
        with patch(f"{transform_module}.get_request_context") as m:
            m.return_value = MCPRequestContext(mode=MCPRequestMode.TOOLS, tool_allowlist=None)
            yield m

    @pytest.mark.asyncio
    async def test_tools_mode_exposes_real_catalog(self, mock_context: Mock) -> None:
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.TOOLS, tool_allowlist=None
        )
        mcp = self.make_server()

        names = {t.name for t in await mcp.list_tools(run_middleware=False)}

        assert {"add", "greet"} <= names
        assert "execute" not in names

    @pytest.mark.asyncio
    async def test_code_execute_mode_collapses_catalog(self, mock_context: Mock) -> None:
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.CODE_EXECUTE, tool_allowlist=frozenset({"add"})
        )
        mcp = self.make_server()

        names = {t.name for t in await mcp.list_tools(run_middleware=False)}

        assert names == {"search", "get_schema", "execute"}

    @pytest.mark.asyncio
    async def test_tools_mode_filters_catalog_by_allowlist(self, mock_context: Mock) -> None:
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.TOOLS, tool_allowlist=frozenset({"add"})
        )
        mcp = self.make_server()

        names = {t.name for t in await mcp.list_tools(run_middleware=False)}
        assert names == {"add"}

    @pytest.mark.asyncio
    async def test_allowlist_tool_names_are_case_sensitive(self, mock_context: Mock) -> None:
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.TOOLS, tool_allowlist=frozenset({"Greet"})
        )
        mcp = self.make_server()

        names = {t.name for t in await mcp.list_tools(run_middleware=False)}
        assert names == set()

        assert await mcp.get_tool("greet") is None

    @pytest.mark.asyncio
    async def test_get_tool_blocked_when_not_in_allowlist(self, mock_context: Mock) -> None:
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.TOOLS, tool_allowlist=frozenset({"add"})
        )
        mcp = self.make_server()

        assert await mcp.get_tool("add") is not None
        assert await mcp.get_tool("greet") is None

    @pytest.mark.asyncio
    async def test_mode_switch_between_calls(self, mock_context: Mock) -> None:
        mcp = self.make_server()

        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.TOOLS, tool_allowlist=None
        )
        tools_view = {t.name for t in await mcp.list_tools(run_middleware=False)}

        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.CODE_EXECUTE, tool_allowlist=None
        )
        code_view = {t.name for t in await mcp.list_tools(run_middleware=False)}
        assert code_view == {"search", "get_schema", "execute"}

        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.TOOLS, tool_allowlist=None
        )
        back_view = {t.name for t in await mcp.list_tools(run_middleware=False)}
        assert back_view == tools_view


class _NamedTool:
    def __init__(self, name: str) -> None:
        self.name = name


class TestFilterToolsByAllowlist:
    def test_skips_unknown_allowlist_names(self) -> None:
        tools = [_NamedTool("add")]
        result = filter_tools_by_allowlist(tools, frozenset({"add", "missing"}))  # type: ignore[arg-type]
        assert [t.name for t in result] == ["add"]

    def test_is_tool_name_allowed_requires_exact_name(self) -> None:
        assert is_tool_name_allowed("add", frozenset({"add"}))
        assert not is_tool_name_allowed("Add", frozenset({"add"}))
