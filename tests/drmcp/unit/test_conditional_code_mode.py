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

"""Tests for ``ConditionalCodeMode`` — per-request CodeMode gating."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp import FastMCP
from fastmcp.experimental.transforms.code_mode import _ensure_async

from datarobot_genai.drmcp.core.conditional_code_mode import ConditionalCodeMode
from datarobot_genai.drtools.core.mode import MCPMode


class _UnsafeTestSandboxProvider:
    """Test-only sandbox that uses exec(). Never use in production."""

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


def _make_server() -> FastMCP:
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
def module_under_test() -> str:
    return "datarobot_genai.drmcp.core.conditional_code_mode"


@pytest.fixture
def mock_get_mcp_mode_from_headers(module_under_test) -> Iterator[Mock]:
    with patch(f"{module_under_test}.get_mcp_mode_from_headers") as m:
        m.return_value = MCPMode.TOOLS
        yield m


@pytest.mark.asyncio
async def test_tools_mode_exposes_real_catalog(mock_get_mcp_mode_from_headers: Any) -> None:
    """No header (default TOOLS) → real tools visible, CodeMode meta-tools hidden."""
    mock_get_mcp_mode_from_headers.return_value = MCPMode.TOOLS
    mcp = _make_server()

    listed = await mcp.list_tools(run_middleware=False)
    names = {t.name for t in listed}

    assert {"add", "greet"} <= names
    assert "execute" not in names
    assert "search" not in names
    assert "get_schema" not in names


@pytest.mark.asyncio
async def test_code_execute_mode_collapses_catalog(mock_get_mcp_mode_from_headers: Any) -> None:
    """code_execute → catalog collapses to search + get_schema + execute."""
    mock_get_mcp_mode_from_headers.return_value = MCPMode.CODE_EXECUTE
    mcp = _make_server()

    listed = await mcp.list_tools(run_middleware=False)
    names = {t.name for t in listed}

    assert names == {"search", "get_schema", "execute"}


@pytest.mark.asyncio
async def test_mode_switch_between_calls(mock_get_mcp_mode_from_headers: Any) -> None:
    """The same server instance returns different catalogs as the mode changes."""
    mcp = _make_server()

    mock_get_mcp_mode_from_headers.return_value = MCPMode.TOOLS
    tools_view = {t.name for t in await mcp.list_tools(run_middleware=False)}
    assert {"add", "greet"} <= tools_view

    mock_get_mcp_mode_from_headers.return_value = MCPMode.CODE_EXECUTE
    code_view = {t.name for t in await mcp.list_tools(run_middleware=False)}
    assert code_view == {"search", "get_schema", "execute"}

    mock_get_mcp_mode_from_headers.return_value = MCPMode.TOOLS
    back_view = {t.name for t in await mcp.list_tools(run_middleware=False)}
    assert back_view == tools_view


@pytest.mark.asyncio
async def test_get_tool_passes_through_in_tools_mode(mock_get_mcp_mode_from_headers: Any) -> None:
    """Tools mode: get_tool resolves real tools and NOT CodeMode meta-tools."""
    mock_get_mcp_mode_from_headers.return_value = MCPMode.TOOLS
    mcp = _make_server()

    add_tool = await mcp.get_tool("add")
    assert add_tool is not None
    assert add_tool.name == "add"

    # Meta-tools are not addressable in tools mode
    assert await mcp.get_tool("execute") is None
    assert await mcp.get_tool("search") is None


@pytest.mark.asyncio
async def test_get_tool_resolves_meta_tools_in_code_execute_mode(
    mock_get_mcp_mode_from_headers: Any,
) -> None:
    """code_execute mode: get_tool resolves CodeMode meta-tools by name."""
    mock_get_mcp_mode_from_headers.return_value = MCPMode.CODE_EXECUTE
    mcp = _make_server()

    for name in ("search", "get_schema", "execute"):
        tool = await mcp.get_tool(name)
        assert tool is not None, f"Expected {name} to be resolvable in code_execute mode"
        assert tool.name == name


@pytest.mark.asyncio
async def test_no_header_defaults_to_tools_mode() -> None:
    """When fastmcp's request context yields no header, defaults to TOOLS."""
    mcp = _make_server()

    # Don't patch get_mcp_mode — let the real implementation run with no HTTP ctx.
    # _get_http_headers returns {} outside a request, so get_mcp_mode → TOOLS.
    listed = await mcp.list_tools(run_middleware=False)
    names = {t.name for t in listed}
    assert {"add", "greet"} <= names
    assert "execute" not in names
