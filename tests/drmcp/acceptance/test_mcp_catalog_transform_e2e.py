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

"""End-to-end tests for MCP catalog transforms against a live HTTP MCP server."""

import pytest

from datarobot_genai.drmcp.test_utils.mcp_utils_ete import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.mcp_utils_ete import get_dr_mcp_server_url
from tests.drmcp.helpers.mcp_catalog_transform import catalog_transform_headers
from tests.drmcp.helpers.mcp_catalog_transform import tool_names_from_list_tools_result

pytestmark = pytest.mark.skipif(
    not get_dr_mcp_server_url(),
    reason="DR_MCP_SERVER_URL must point at a running MCP server",
)


@pytest.mark.asyncio
class TestMcpCatalogTransformE2E:
    """Full client/server path: discover tools under an allowlist, then invoke one."""

    async def test_list_then_call_allowed_tool(self) -> None:
        async with ete_test_mcp_session() as session:
            baseline = tool_names_from_list_tools_result(await session.list_tools())
        assert baseline

        allowed = sorted(baseline)[0]
        headers = catalog_transform_headers(tools=allowed)

        async with ete_test_mcp_session(additional_headers=headers) as session:
            listed = tool_names_from_list_tools_result(await session.list_tools())
            assert listed == {allowed}

            result = await session.call_tool(allowed, {})
            assert not result.isError, (
                result.content[0].text if result.content else "call_tool failed"
            )

    async def test_mixed_case_header_key_still_applies_filter(self) -> None:
        async with ete_test_mcp_session() as session:
            baseline = tool_names_from_list_tools_result(await session.list_tools())
        sample = sorted(baseline)[0]

        headers = {"X-DataRobot-MCP-Tools": sample}
        async with ete_test_mcp_session(additional_headers=headers) as session:
            names = tool_names_from_list_tools_result(await session.list_tools())
        assert names == {sample}
