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

from datarobot_genai.drmcpbase.dynamic_tools.enums import DataRobotMCPToolCategory
from datarobot_genai.drmcpbase.fastmcp_transforms import register_mcp_catalog_transform
from datarobot_genai.drmcpbase.fastmcp_transforms.transform import DataRobotMCPCatalogTransform
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCP_ENABLE_DYNAMIC_TOOLS_HEADER
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCP_ENABLE_PROXY_HEADER
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCP_TOOLSETS_HEADER
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCPRequestContext
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCPRequestMode
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import _request_context_cache
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import _resolve_toolsets
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import filter_tools_by_allowlist
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import get_header_case_insensitive
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import get_request_context
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import is_category_disabled_for_request
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import is_tool_name_allowed
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import parse_bool_header
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import parse_disabled_categories
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


class TestParseBoolHeader:
    def test_default_true_when_absent(self) -> None:
        # GIVEN no header value / WHEN parsed / THEN the default (true) applies
        assert parse_bool_header(None) is True

    def test_default_false_when_absent_and_default_false(self) -> None:
        assert parse_bool_header(None, default=False) is False

    @pytest.mark.parametrize("raw", ["false", "FALSE", "False", " false ", "0", "no", "off"])
    def test_false_variants(self, raw: str) -> None:
        assert parse_bool_header(raw) is False

    @pytest.mark.parametrize("raw", ["true", "TRUE", " true ", "1", "yes", "on"])
    def test_true_variants(self, raw: str) -> None:
        assert parse_bool_header(raw, default=False) is True

    @pytest.mark.parametrize("raw", ["", "  ", "banana", "flase"])
    def test_unrecognized_values_fall_back_to_default(self, raw: str) -> None:
        assert parse_bool_header(raw) is True
        assert parse_bool_header(raw, default=False) is False


class TestParseDisabledCategories:
    def test_empty_when_no_gate_headers(self) -> None:
        assert parse_disabled_categories({}) == frozenset()

    def test_explicit_true_disables_nothing(self) -> None:
        headers = {MCP_ENABLE_PROXY_HEADER: "true", MCP_ENABLE_DYNAMIC_TOOLS_HEADER: "true"}
        assert parse_disabled_categories(headers) == frozenset()

    def test_proxy_gate_false_disables_proxied_user_mcp(self) -> None:
        headers = {MCP_ENABLE_PROXY_HEADER: "false"}
        assert parse_disabled_categories(headers) == frozenset(
            {DataRobotMCPToolCategory.PROXIED_USER_MCP.name}
        )

    def test_dynamic_tools_gate_false_disables_user_tool_deployment(self) -> None:
        headers = {MCP_ENABLE_DYNAMIC_TOOLS_HEADER: "false"}
        assert parse_disabled_categories(headers) == frozenset(
            {DataRobotMCPToolCategory.USER_TOOL_DEPLOYMENT.name}
        )

    def test_both_gates_false_disables_both_categories(self) -> None:
        headers = {
            MCP_ENABLE_PROXY_HEADER: "false",
            MCP_ENABLE_DYNAMIC_TOOLS_HEADER: "false",
        }
        assert parse_disabled_categories(headers) == frozenset(
            {
                DataRobotMCPToolCategory.PROXIED_USER_MCP.name,
                DataRobotMCPToolCategory.USER_TOOL_DEPLOYMENT.name,
            }
        )

    def test_mixed_case_header_key_is_recognized(self) -> None:
        headers = {"X-DataRobot-MCP-Enable-Proxy": "false"}
        assert parse_disabled_categories(headers) == frozenset(
            {DataRobotMCPToolCategory.PROXIED_USER_MCP.name}
        )


class TestMCPRequestContextCategoryGates:
    def test_default_context_has_no_disabled_categories(self) -> None:
        ctx = MCPRequestContext.from_headers({})
        assert ctx.disabled_categories == frozenset()

    def test_gate_header_populates_disabled_categories(self) -> None:
        ctx = MCPRequestContext.from_headers({MCP_ENABLE_PROXY_HEADER: "false"})
        assert ctx.disabled_categories == frozenset(
            {DataRobotMCPToolCategory.PROXIED_USER_MCP.name}
        )

    def test_gates_are_independent_of_mode_and_allowlist(self) -> None:
        ctx = MCPRequestContext.from_headers(
            {
                MCP_ENABLE_DYNAMIC_TOOLS_HEADER: "false",
                "x-datarobot-mcp-mode": "code",
                "x-datarobot-mcp-tools": "add",
            }
        )
        assert ctx.mode == MCPRequestMode.CODE
        assert ctx.tool_allowlist == frozenset({"add"})
        assert ctx.disabled_categories == frozenset(
            {DataRobotMCPToolCategory.USER_TOOL_DEPLOYMENT.name}
        )


class TestIsCategoryDisabledForRequest:
    def test_false_when_gate_absent(self, utils_module: str) -> None:
        with patch(f"{utils_module}.get_fast_mcp_http_headers", return_value={}):
            assert not is_category_disabled_for_request(
                DataRobotMCPToolCategory.PROXIED_USER_MCP.name
            )

    def test_true_when_gate_header_false(self, utils_module: str) -> None:
        with patch(
            f"{utils_module}.get_fast_mcp_http_headers",
            return_value={MCP_ENABLE_PROXY_HEADER: "false"},
        ):
            assert is_category_disabled_for_request(DataRobotMCPToolCategory.PROXIED_USER_MCP.name)

    def test_false_when_request_context_unavailable(self, utils_module: str) -> None:
        # GIVEN no HTTP request context (e.g. startup retrospection)
        with patch(
            f"{utils_module}.get_fast_mcp_http_headers",
            side_effect=RuntimeError("no request"),
        ):
            # WHEN the gate is consulted / THEN it fails open to "not disabled"
            assert not is_category_disabled_for_request(
                DataRobotMCPToolCategory.PROXIED_USER_MCP.name
            )


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

    @pytest.mark.parametrize("code", ["code", "CODE"])
    def test_returns_code_when_header_set(self, code: str, mock_get_fast_mcp_headers: Mock) -> None:
        mock_get_fast_mcp_headers.return_value = {"x-datarobot-mcp-mode": code}
        assert MCPRequestContext.from_current_http_request().mode == MCPRequestMode.CODE

    @pytest.mark.parametrize("search", ["search", "SEARCH"])
    def test_returns_search_when_header_set(
        self, search: str, mock_get_fast_mcp_headers: Mock
    ) -> None:
        mock_get_fast_mcp_headers.return_value = {"x-datarobot-mcp-mode": search}
        assert MCPRequestContext.from_current_http_request().mode == MCPRequestMode.SEARCH

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
    async def test_code_mode_collapses_catalog(self, mock_context: Mock) -> None:
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.CODE, tool_allowlist=frozenset({"add"})
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

        mock_context.return_value = MCPRequestContext(mode=MCPRequestMode.CODE, tool_allowlist=None)
        code_view = {t.name for t in await mcp.list_tools(run_middleware=False)}
        assert code_view == {"search", "get_schema", "execute"}

        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.TOOLS, tool_allowlist=None
        )
        back_view = {t.name for t in await mcp.list_tools(run_middleware=False)}
        assert back_view == tools_view


class TestCategoryGatesInTransform:
    """Category gates (Track 0) — precedence: gates → mode → allowlist."""

    PROXIED = DataRobotMCPToolCategory.PROXIED_USER_MCP.name
    DYNAMIC = DataRobotMCPToolCategory.USER_TOOL_DEPLOYMENT.name

    @staticmethod
    def make_server() -> FastMCP:
        mcp = FastMCP("category gates test")

        @mcp.tool
        def add(a: int, b: int) -> int:
            """Add two numbers (built-in, untagged)."""
            return a + b

        @mcp.tool(meta={"tool_category": DataRobotMCPToolCategory.PROXIED_USER_MCP.name})
        def proxied_tool() -> str:
            """Return a canned value (stands in for a proxied user-MCP tool)."""
            return "proxied"

        @mcp.tool(meta={"tool_category": DataRobotMCPToolCategory.USER_TOOL_DEPLOYMENT.name})
        def dynamic_tool() -> str:
            """Return a canned value (stands in for a dynamic deployment tool)."""
            return "dynamic"

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
    async def test_gates_default_on_all_tools_visible(self, mock_context: Mock) -> None:
        # GIVEN no gate headers (defaults: enabled)
        mcp = self.make_server()

        names = {t.name for t in await mcp.list_tools(run_middleware=False)}

        assert {"add", "proxied_tool", "dynamic_tool"} <= names

    @pytest.mark.asyncio
    async def test_proxy_gate_hides_proxied_tools_only(self, mock_context: Mock) -> None:
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.TOOLS,
            tool_allowlist=None,
            disabled_categories=frozenset({self.PROXIED}),
        )
        mcp = self.make_server()

        names = {t.name for t in await mcp.list_tools(run_middleware=False)}

        assert "proxied_tool" not in names
        assert {"add", "dynamic_tool"} <= names

    @pytest.mark.asyncio
    async def test_dynamic_gate_hides_deployment_tools_only(self, mock_context: Mock) -> None:
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.TOOLS,
            tool_allowlist=None,
            disabled_categories=frozenset({self.DYNAMIC}),
        )
        mcp = self.make_server()

        names = {t.name for t in await mcp.list_tools(run_middleware=False)}

        assert "dynamic_tool" not in names
        assert {"add", "proxied_tool"} <= names

    @pytest.mark.asyncio
    async def test_gate_takes_precedence_over_allowlist(self, mock_context: Mock) -> None:
        # GIVEN the proxied tool is explicitly allowlisted but its category is gated off
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.TOOLS,
            tool_allowlist=frozenset({"proxied_tool", "add"}),
            disabled_categories=frozenset({self.PROXIED}),
        )
        mcp = self.make_server()

        # WHEN listing / THEN the gated tool stays hidden despite the allowlist
        names = {t.name for t in await mcp.list_tools(run_middleware=False)}
        assert names == {"add"}

    @pytest.mark.asyncio
    async def test_gated_tool_is_not_resolvable(self, mock_context: Mock) -> None:
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.TOOLS,
            tool_allowlist=None,
            disabled_categories=frozenset({self.PROXIED}),
        )
        mcp = self.make_server()

        assert await mcp.get_tool("proxied_tool") is None
        assert await mcp.get_tool("add") is not None

    @pytest.mark.asyncio
    async def test_gate_applies_in_code_mode_get_tool(self, mock_context: Mock) -> None:
        # GIVEN code mode with the proxy category gated off
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.CODE,
            tool_allowlist=None,
            disabled_categories=frozenset({self.PROXIED}),
        )
        mcp = self.make_server()

        # THEN the gated tool cannot be resolved even in code mode,
        # while the synthetic discovery tools stay available (no category meta)
        assert await mcp.get_tool("proxied_tool") is None
        assert await mcp.get_tool("execute") is not None

    @pytest.mark.asyncio
    async def test_gate_off_then_on_between_requests(self, mock_context: Mock) -> None:
        mcp = self.make_server()

        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.TOOLS,
            tool_allowlist=None,
            disabled_categories=frozenset({self.PROXIED}),
        )
        gated = {t.name for t in await mcp.list_tools(run_middleware=False)}
        assert "proxied_tool" not in gated

        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.TOOLS, tool_allowlist=None
        )
        ungated = {t.name for t in await mcp.list_tools(run_middleware=False)}
        assert "proxied_tool" in ungated


class TestSearchMode:
    """`x-datarobot-mcp-mode: search` — catalog collapses to tool_search + call_tool."""

    PROXIED = DataRobotMCPToolCategory.PROXIED_USER_MCP.name

    @staticmethod
    def make_server() -> FastMCP:
        mcp = FastMCP("search mode test")

        @mcp.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        @mcp.tool
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        @mcp.tool(meta={"tool_category": DataRobotMCPToolCategory.PROXIED_USER_MCP.name})
        def proxied_tool() -> str:
            """Return a canned proxied value."""
            return "proxied"

        mcp.add_transform(
            DataRobotMCPCatalogTransform(sandbox_provider=_UnsafeTestSandboxProvider())
        )
        return mcp

    @pytest.fixture
    def mock_context(self, transform_module: str) -> Iterator[Mock]:
        with patch(f"{transform_module}.get_request_context") as m:
            m.return_value = MCPRequestContext(mode=MCPRequestMode.SEARCH, tool_allowlist=None)
            yield m

    @pytest.mark.asyncio
    async def test_listing_collapses_to_search_interface(self, mock_context: Mock) -> None:
        mcp = self.make_server()

        names = {t.name for t in await mcp.list_tools(run_middleware=False)}

        assert names == {"tool_search", "call_tool"}

    @pytest.mark.asyncio
    async def test_allowlisted_tools_stay_pinned_in_listing(self, mock_context: Mock) -> None:
        # GIVEN a client that re-lists with x-datarobot-mcp-tools=<found names>
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.SEARCH, tool_allowlist=frozenset({"add"})
        )
        mcp = self.make_server()

        # THEN the allowlisted tool's full definition rides along with the interface
        names = {t.name for t in await mcp.list_tools(run_middleware=False)}
        assert names == {"add", "tool_search", "call_tool"}

    @pytest.mark.asyncio
    async def test_tool_search_finds_matching_tools(self, mock_context: Mock) -> None:
        mcp = self.make_server()

        result = await mcp.call_tool("tool_search", {"query": "add two numbers"})

        text = result.content[0].text
        assert "add" in text

    @pytest.mark.asyncio
    async def test_tool_search_respects_category_gates(self, mock_context: Mock) -> None:
        # GIVEN the proxied category is gated off for this request
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.SEARCH,
            tool_allowlist=None,
            disabled_categories=frozenset({self.PROXIED}),
        )
        mcp = self.make_server()

        # WHEN searching with a query that would match the gated tool
        result = await mcp.call_tool("tool_search", {"query": "canned proxied value"})

        # THEN the gated tool does not appear in the results
        assert "proxied_tool" not in result.content[0].text

    @pytest.mark.asyncio
    async def test_tool_search_respects_allowlist_cap(self, mock_context: Mock) -> None:
        # GIVEN an allowlist that excludes `greet`
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.SEARCH, tool_allowlist=frozenset({"add"})
        )
        mcp = self.make_server()

        result = await mcp.call_tool("tool_search", {"query": "say hello greeting"})

        assert "greet" not in result.content[0].text

    @pytest.mark.asyncio
    async def test_call_tool_proxy_executes_discovered_tool(self, mock_context: Mock) -> None:
        mcp = self.make_server()

        result = await mcp.call_tool("call_tool", {"name": "add", "arguments": {"a": 1, "b": 2}})

        assert "3" in result.content[0].text

    @pytest.mark.asyncio
    async def test_call_tool_proxy_rejects_synthetic_names(self, mock_context: Mock) -> None:
        mcp = self.make_server()

        with pytest.raises(Exception, match="synthetic"):
            await mcp.call_tool("call_tool", {"name": "tool_search", "arguments": {}})

    @pytest.mark.asyncio
    async def test_hidden_tools_remain_resolvable_without_allowlist(
        self, mock_context: Mock
    ) -> None:
        # Hidden-but-callable is the search-mode contract: tool_search returns
        # names that must be directly callable (no dangling references).
        mcp = self.make_server()

        assert await mcp.get_tool("tool_search") is not None
        assert await mcp.get_tool("call_tool") is not None
        assert await mcp.get_tool("greet") is not None

    @pytest.mark.asyncio
    async def test_gates_and_allowlist_cap_resolution(self, mock_context: Mock) -> None:
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.SEARCH,
            tool_allowlist=frozenset({"add"}),
            disabled_categories=frozenset({self.PROXIED}),
        )
        mcp = self.make_server()

        # Synthetic interface tools stay resolvable despite the allowlist
        assert await mcp.get_tool("tool_search") is not None
        assert await mcp.get_tool("call_tool") is not None
        assert await mcp.get_tool("add") is not None
        # Catalog tools outside the allowlist / in a gated category do not resolve
        assert await mcp.get_tool("greet") is None
        assert await mcp.get_tool("proxied_tool") is None

    @pytest.mark.asyncio
    async def test_call_tool_proxy_blocked_for_gated_tool(self, mock_context: Mock) -> None:
        # GIVEN the proxied category is gated off
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.SEARCH,
            tool_allowlist=None,
            disabled_categories=frozenset({self.PROXIED}),
        )
        mcp = self.make_server()

        # THEN the proxy cannot be used to reach the gated tool
        with pytest.raises(Exception, match="proxied_tool"):
            await mcp.call_tool("call_tool", {"name": "proxied_tool", "arguments": {}})


class TestCodeModeAllowlistEnforcement:
    """H5 regression: code mode must honor the tool allowlist."""

    @staticmethod
    def make_server() -> FastMCP:
        return TestSearchMode.make_server()

    @pytest.fixture
    def mock_context(self, transform_module: str) -> Iterator[Mock]:
        with patch(f"{transform_module}.get_request_context") as m:
            m.return_value = MCPRequestContext(
                mode=MCPRequestMode.CODE, tool_allowlist=frozenset({"add"})
            )
            yield m

    @pytest.mark.asyncio
    async def test_get_tool_blocked_when_not_in_allowlist(self, mock_context: Mock) -> None:
        # GIVEN code mode with an allowlist
        mcp = self.make_server()

        # THEN a non-allowlisted tool is not resolvable — switching the mode
        # header no longer escapes the allowlist (regression: it used to)
        assert await mcp.get_tool("greet") is None
        assert await mcp.get_tool("add") is not None

    @pytest.mark.asyncio
    async def test_synthetic_mode_tools_stay_resolvable(self, mock_context: Mock) -> None:
        mcp = self.make_server()

        assert await mcp.get_tool("execute") is not None
        assert await mcp.get_tool("search") is not None
        assert await mcp.get_tool("get_schema") is not None

    @pytest.mark.asyncio
    async def test_execute_cannot_call_non_allowlisted_tool(self, mock_context: Mock) -> None:
        mcp = self.make_server()

        with pytest.raises(Exception, match="greet"):
            await mcp.call_tool(
                "execute",
                {"code": 'return await call_tool("greet", {"name": "x"})'},
            )

    @pytest.mark.asyncio
    async def test_discovery_search_does_not_leak_non_allowlisted_tools(
        self, mock_context: Mock
    ) -> None:
        # GIVEN code-mode discovery reads the catalog through get_tool_catalog,
        # which bypasses transform_tools (regression: gates/allowlist were skipped)
        mcp = self.make_server()

        result = await mcp.call_tool("search", {"query": "say hello greeting"})

        assert "greet" not in result.content[0].text

    @pytest.mark.asyncio
    async def test_discovery_search_does_not_leak_gated_tools(self, mock_context: Mock) -> None:
        mock_context.return_value = MCPRequestContext(
            mode=MCPRequestMode.CODE,
            tool_allowlist=None,
            disabled_categories=frozenset({DataRobotMCPToolCategory.PROXIED_USER_MCP.name}),
        )
        mcp = self.make_server()

        result = await mcp.call_tool("search", {"query": "canned proxied value"})

        assert "proxied_tool" not in result.content[0].text


class _NamedTool:
    def __init__(self, name: str) -> None:
        self.name = name


class TestResolveToolsets:
    def test_always_returns_empty_frozenset(self) -> None:
        assert _resolve_toolsets(None) == frozenset()

    def test_non_empty_raw_still_returns_empty(self) -> None:
        # The stub ignores the header value — implementation is for global-mcp only.
        assert _resolve_toolsets("my_toolset") == frozenset()

    def test_returns_frozenset_type(self) -> None:
        result = _resolve_toolsets("anything")
        assert isinstance(result, frozenset)


class TestMCPRequestContextFromHeaders:
    def test_no_headers_gives_tools_mode_no_allowlist(self) -> None:
        ctx = MCPRequestContext.from_headers({})
        assert ctx.mode == MCPRequestMode.TOOLS
        assert ctx.tool_allowlist is None

    def test_tools_header_sets_allowlist(self) -> None:
        ctx = MCPRequestContext.from_headers({"x-datarobot-mcp-tools": "jira_search_issues"})
        assert ctx.tool_allowlist is not None
        assert "jira_search_issues" in ctx.tool_allowlist

    def test_toolsets_header_alone_returns_empty_allowlist(self) -> None:
        # Stub always resolves to empty frozenset; empty frozenset is falsy →
        # combined falls through to tools_allowlist (None).
        ctx = MCPRequestContext.from_headers({MCP_TOOLSETS_HEADER: "my_toolset"})
        assert ctx.tool_allowlist is None

    def test_both_headers_unions_results(self) -> None:
        # Patch _resolve_toolsets so it returns a non-empty set to exercise the union branch.
        with patch(
            "datarobot_genai.drmcpbase.fastmcp_transforms.utils._resolve_toolsets",
            return_value=frozenset({"extra_tool"}),
        ):
            ctx = MCPRequestContext.from_headers({"x-datarobot-mcp-tools": "jira_search_issues"})
        assert ctx.tool_allowlist is not None
        assert "jira_search_issues" in ctx.tool_allowlist
        assert "extra_tool" in ctx.tool_allowlist

    def test_only_toolsets_non_empty_becomes_allowlist(self) -> None:
        # If tools header absent but toolsets resolves non-empty, it becomes the allowlist.
        with patch(
            "datarobot_genai.drmcpbase.fastmcp_transforms.utils._resolve_toolsets",
            return_value=frozenset({"toolset_tool"}),
        ):
            ctx = MCPRequestContext.from_headers({})
        assert ctx.tool_allowlist == frozenset({"toolset_tool"})

    def test_category_header_expands_to_tool_names(self) -> None:
        ctx = MCPRequestContext.from_headers({"x-datarobot-mcp-tools": "dr_connector_jira"})
        assert ctx.tool_allowlist is not None
        assert "jira_search_issues" in ctx.tool_allowlist
        assert "jira_get_issue" in ctx.tool_allowlist
        # Non-jira tools must NOT be in the allowlist
        assert "confluence_get_page" not in ctx.tool_allowlist


class TestGetRequestContextCaching:
    def test_first_call_builds_and_caches_context(self, utils_module: str) -> None:
        with patch(f"{utils_module}.get_fast_mcp_http_headers", return_value={}) as mock_headers:
            ctx1 = get_request_context()
            ctx2 = get_request_context()
        assert ctx1 is ctx2
        # Headers should only be fetched once; second call hits the cache.
        mock_headers.assert_called_once()

    def test_cache_reset_yields_fresh_context(self, utils_module: str) -> None:
        with patch(f"{utils_module}.get_fast_mcp_http_headers", return_value={}):
            ctx1 = get_request_context()
        _request_context_cache.set(None)
        with patch(
            f"{utils_module}.get_fast_mcp_http_headers",
            return_value={"x-datarobot-mcp-tools": "jira_search_issues"},
        ):
            ctx2 = get_request_context()
        assert ctx1 is not ctx2
        assert ctx2.tool_allowlist is not None
        assert "jira_search_issues" in ctx2.tool_allowlist


class TestMCPToolsetsHeaderConstant:
    def test_toolsets_header_name(self) -> None:
        assert MCP_TOOLSETS_HEADER == "x-datarobot-mcp-toolsets"


class TestFilterToolsByAllowlist:
    def test_skips_unknown_allowlist_names(self) -> None:
        tools = [_NamedTool("add")]
        result = filter_tools_by_allowlist(tools, frozenset({"add", "missing"}))  # type: ignore[arg-type]
        assert [t.name for t in result] == ["add"]

    def test_is_tool_name_allowed_requires_exact_name(self) -> None:
        assert is_tool_name_allowed("add", frozenset({"add"}))
        assert not is_tool_name_allowed("Add", frozenset({"add"}))
