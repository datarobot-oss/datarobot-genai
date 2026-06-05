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

"""Integration tests for per-request MCP catalog transforms.

Uses an in-process DataRobot MCP server (full ``create_mcp_server`` wiring) with
patched HTTP headers. Stdio subprocess sessions cannot forward ``x-datarobot-*``
headers; see ``test_stdio_session_lists_tools_without_header_filter`` for that path.
"""

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from fastmcp import FastMCP
from fastmcp.experimental.transforms.code_mode import _ensure_async

# Registers get_auth_context_user_info on the global MCP instance.
import tests.drmcp.acceptance.test_tools  # noqa: F401
from datarobot_genai.drmcp import create_mcp_server
from datarobot_genai.drmcp import integration_test_mcp_session
from datarobot_genai.drmcpbase.fastmcp_transforms.transform import DataRobotMCPCatalogTransform
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import _request_context_cache
from tests.drmcp.helpers.mcp_catalog_transform import CODE_EXECUTE_TOOL_NAMES
from tests.drmcp.helpers.mcp_catalog_transform import catalog_transform_headers
from tests.drmcp.helpers.mcp_catalog_transform import tool_names_from_list_tools_result

UTILS_MODULE = "datarobot_genai.drmcpbase.fastmcp_transforms.utils"


@pytest.fixture(autouse=True)
def _reset_request_context_cache() -> Iterator[None]:
    _request_context_cache.set(None)
    yield
    _request_context_cache.set(None)


@pytest.fixture(scope="module")
def in_process_drmcp_mcp() -> FastMCP:
    server = create_mcp_server(transport="stdio", load_native_mcp_tools=False)
    return server._mcp


class _UnsafeTestSandboxProvider:
    async def run(
        self,
        code: str,
        *,
        inputs: dict | None = None,
        external_functions: dict | None = None,
    ) -> object:
        namespace: dict = {}
        if inputs:
            namespace.update(inputs)
        if external_functions:
            namespace.update({k: _ensure_async(v) for k, v in external_functions.items()})
        wrapped = "async def __test_main__():\n"
        for line in code.splitlines():
            wrapped += f"    {line}\n" if line.strip() else "    pass\n"
        exec(wrapped, namespace)  # noqa: S102 — test-only
        return await namespace["__test_main__"]()


@pytest.fixture(scope="module")
def minimal_catalog_mcp() -> FastMCP:
    mcp = FastMCP("mcp catalog transform integration")

    @mcp.tool
    def alpha() -> str:
        return "alpha"

    @mcp.tool
    def beta() -> str:
        return "beta"

    mcp.add_transform(DataRobotMCPCatalogTransform(sandbox_provider=_UnsafeTestSandboxProvider()))
    return mcp


@pytest.mark.asyncio
class TestMcpCatalogTransformInProcess:
    async def test_no_header_lists_all_tools(self, minimal_catalog_mcp: FastMCP) -> None:
        with patch(f"{UTILS_MODULE}.get_fast_mcp_http_headers", return_value={}):
            names = {t.name for t in await minimal_catalog_mcp.list_tools(run_middleware=False)}
        assert names == {"alpha", "beta"}

    async def test_tools_header_filters_list_and_get_tool(
        self, minimal_catalog_mcp: FastMCP
    ) -> None:
        headers = catalog_transform_headers(tools="alpha")
        with patch(f"{UTILS_MODULE}.get_fast_mcp_http_headers", return_value=headers):
            listed = await minimal_catalog_mcp.list_tools(run_middleware=False)
            assert {t.name for t in listed} == {"alpha"}
            assert await minimal_catalog_mcp.get_tool("beta") is None
            assert (await minimal_catalog_mcp.get_tool("alpha")) is not None

    async def test_unknown_allowlist_names_return_empty_catalog(
        self, minimal_catalog_mcp: FastMCP
    ) -> None:
        headers = catalog_transform_headers(tools="missing,also_missing")
        with patch(f"{UTILS_MODULE}.get_fast_mcp_http_headers", return_value=headers):
            listed = await minimal_catalog_mcp.list_tools(run_middleware=False)
        assert listed == []

    async def test_code_execute_mode_collapses_catalog(self, minimal_catalog_mcp: FastMCP) -> None:
        headers = catalog_transform_headers(mode="code_execute", tools="alpha")
        with patch(f"{UTILS_MODULE}.get_fast_mcp_http_headers", return_value=headers):
            names = {t.name for t in await minimal_catalog_mcp.list_tools(run_middleware=False)}
        assert names == CODE_EXECUTE_TOOL_NAMES

    async def test_drmcp_server_registers_transform(self, in_process_drmcp_mcp: FastMCP) -> None:
        headers = catalog_transform_headers(tools="get_auth_context_user_info")
        with patch(f"{UTILS_MODULE}.get_fast_mcp_http_headers", return_value=headers):
            names = {t.name for t in await in_process_drmcp_mcp.list_tools(run_middleware=False)}
        assert names == {"get_auth_context_user_info"}


@pytest.mark.asyncio
class TestMcpCatalogTransformStdioSession:
    async def test_stdio_session_lists_tools_without_header_filter(self) -> None:
        async with integration_test_mcp_session() as session:
            names = tool_names_from_list_tools_result(await session.list_tools())
        assert "get_user_greeting" in names
