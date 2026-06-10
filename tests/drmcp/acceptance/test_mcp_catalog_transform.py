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

"""Acceptance tests for MCP catalog transforms over streamable HTTP."""

import pytest

from datarobot_genai.drmcp.test_utils.mcp_utils_ete import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.mcp_utils_ete import get_dr_mcp_server_url
from tests.drmcp.helpers.mcp_catalog_transform import CODE_EXECUTE_TOOL_NAMES
from tests.drmcp.helpers.mcp_catalog_transform import catalog_transform_headers
from tests.drmcp.helpers.mcp_catalog_transform import tool_names_from_list_tools_result

pytestmark = pytest.mark.skipif(
    not get_dr_mcp_server_url(),
    reason="DR_MCP_SERVER_URL must point at a running MCP server",
)


@pytest.mark.asyncio
class TestMcpCatalogTransformAcceptance:
    async def test_list_tools_without_filter_returns_broad_catalog(self) -> None:
        async with ete_test_mcp_session() as session:
            names = tool_names_from_list_tools_result(await session.list_tools())
        assert len(names) > 1

    async def test_tools_header_filters_listed_tools(self) -> None:
        async with ete_test_mcp_session() as session:
            baseline = tool_names_from_list_tools_result(await session.list_tools())
        assert baseline

        sample = sorted(baseline)[0]
        headers = catalog_transform_headers(tools=sample)

        async with ete_test_mcp_session(additional_headers=headers) as session:
            names = tool_names_from_list_tools_result(await session.list_tools())

        assert names == {sample}

    async def test_tools_header_is_case_sensitive_for_tool_names(self) -> None:
        async with ete_test_mcp_session() as session:
            baseline = tool_names_from_list_tools_result(await session.list_tools())
        sample = sorted(baseline)[0]
        wrong_case = sample.swapcase()
        if wrong_case == sample:
            pytest.skip("Tool name is not mixed case")

        headers = catalog_transform_headers(tools=wrong_case)
        async with ete_test_mcp_session(additional_headers=headers) as session:
            names = tool_names_from_list_tools_result(await session.list_tools())
        assert names == set()

    async def test_unknown_tools_header_entries_yield_empty_catalog(self) -> None:
        headers = catalog_transform_headers(tools="__definitely_not_a_registered_tool__")
        async with ete_test_mcp_session(additional_headers=headers) as session:
            names = tool_names_from_list_tools_result(await session.list_tools())
        assert names == set()

    async def test_code_execute_mode_lists_meta_tools_only(self) -> None:
        headers = catalog_transform_headers(mode="code_execute")
        async with ete_test_mcp_session(additional_headers=headers) as session:
            names = tool_names_from_list_tools_result(await session.list_tools())
        assert names == CODE_EXECUTE_TOOL_NAMES

    async def test_code_execute_mode_ignores_tools_allowlist_header(self) -> None:
        async with ete_test_mcp_session() as session:
            baseline = tool_names_from_list_tools_result(await session.list_tools())
        sample = sorted(baseline)[0]

        headers = catalog_transform_headers(mode="code_execute", tools=sample)
        async with ete_test_mcp_session(additional_headers=headers) as session:
            names = tool_names_from_list_tools_result(await session.list_tools())
        assert names == CODE_EXECUTE_TOOL_NAMES

    async def test_call_tool_blocked_when_not_in_allowlist(self) -> None:
        async with ete_test_mcp_session() as session:
            baseline = tool_names_from_list_tools_result(await session.list_tools())
        names = sorted(baseline)
        if len(names) < 2:
            pytest.skip("Need at least two tools to test allowlist call blocking")
        allowed, blocked = names[0], names[1]

        headers = catalog_transform_headers(tools=allowed)
        async with ete_test_mcp_session(additional_headers=headers) as session:
            blocked_result = await session.call_tool(blocked, {})
            assert blocked_result.isError
