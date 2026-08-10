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

"""``x-datarobot-mcp-tools`` must honour the marker-resolved category buckets.

``dr_user_tools`` and ``dr_dynamic_tools`` name no static tools — membership comes
from each tool's ``meta.tool_category`` marker at request time. They used to expand
to the empty set, and an empty-but-present allowlist is a hard deny, so asking for
"Your own tools" hid every tool on the server.
"""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from datarobot_genai.drmcpbase.fastmcp_transforms import utils as transform_utils
from datarobot_genai.drmcpbase.fastmcp_transforms.transform import DataRobotMCPCatalogTransform
from datarobot_genai.drmcputils.categories import parse_tool_allowlist_header


def _tool(name: str, marker: str | None = None) -> Any:
    """Build a stand-in FastMCP ``Tool`` — the transform reads ``name`` and ``meta``."""
    return SimpleNamespace(name=name, meta={"tool_category": marker} if marker else {}, tags=set())


CATALOG = [
    _tool("jira_get_issue"),  # built-in, static taxonomy
    _tool("my_own_tool", "USER_TOOL"),  # user-authored
    _tool("another_of_mine", "USER_TOOL"),
    _tool("a_deployment", "USER_TOOL_DEPLOYMENT"),  # dynamic
    _tool("proxied_thing", "PROXIED_USER_MCP"),  # proxied: no category at all
]


@pytest.fixture(autouse=True)
def _clear_request_context() -> Iterator[None]:
    """Clear the per-request context cache so it cannot leak between tests."""
    yield
    transform_utils._request_context_cache.set(None)


def seeded(header: str | None) -> DataRobotMCPCatalogTransform:
    """Build a transform driven by the request context *header* would produce.

    Seeded inside the running coroutine rather than from a fixture: a ContextVar
    token belongs to the context that created it, and an async test body runs in a
    copy — resetting it from the fixture's context raises.
    """
    headers = {"x-datarobot-mcp-tools": header} if header else {}
    transform_utils._request_context_cache.set(
        transform_utils.MCPRequestContext.from_headers(headers)
    )
    return DataRobotMCPCatalogTransform()


class TestMarkerBucketsInTheToolsHeader:
    def test_bucket_survives_resolution_instead_of_expanding_to_nothing(self) -> None:
        # GIVEN the header names a marker-resolved bucket
        # THEN it is kept as a literal token for the matcher, NOT expanded to the
        # empty set — which the allowlist would have read as "permit nothing".
        assert parse_tool_allowlist_header("dr_user_tools").buckets == frozenset({"dr_user_tools"})
        assert parse_tool_allowlist_header("dr_dynamic_tools").buckets == frozenset(
            {"dr_dynamic_tools"}
        )

    def test_static_categories_still_expand_to_tool_names(self) -> None:
        resolved = parse_tool_allowlist_header("dr_connector_jira")
        assert "jira_search_issues" in resolved.derived
        assert "dr_connector_jira" not in resolved.derived
        assert not resolved.explicit and not resolved.buckets

    @pytest.mark.asyncio
    async def test_user_tools_bucket_lists_the_user_tools(self) -> None:
        transform = seeded("dr_user_tools")
        listed = sorted(t.name for t in await transform.transform_tools(CATALOG))
        assert listed == ["another_of_mine", "my_own_tool"]

    @pytest.mark.asyncio
    async def test_dynamic_tools_bucket_lists_the_deployment_tools(self) -> None:
        transform = seeded("dr_dynamic_tools")
        listed = sorted(t.name for t in await transform.transform_tools(CATALOG))
        assert listed == ["a_deployment"]

    @pytest.mark.asyncio
    async def test_bucket_unions_with_a_static_category(self) -> None:
        transform = seeded("dr_user_tools,dr_connector_jira")
        listed = sorted(t.name for t in await transform.transform_tools(CATALOG))
        assert listed == ["another_of_mine", "jira_get_issue", "my_own_tool"]

    @pytest.mark.asyncio
    async def test_proxied_tools_are_never_admitted_by_a_bucket(self) -> None:
        # Proxied user-MCP tools carry a marker but no category, so no bucket names them.
        transform = seeded("dr_user_tools,dr_dynamic_tools")
        listed = sorted(t.name for t in await transform.transform_tools(CATALOG))
        assert "proxied_thing" not in listed

    @pytest.mark.asyncio
    async def test_call_time_resolution_matches_the_listing(self) -> None:
        # tools/call goes through get_tool, a separate gate. If it disagreed with
        # tools/list the allowlist would either leak or block what it just offered.
        transform = seeded("dr_user_tools")

        async def call_next(name: str, *, version: Any = None) -> Any:
            return next((t for t in CATALOG if t.name == name), None)

        assert await transform.get_tool("my_own_tool", call_next) is not None
        for denied in ("jira_get_issue", "a_deployment", "proxied_thing"):
            assert await transform.get_tool(denied, call_next) is None, denied

    @pytest.mark.asyncio
    async def test_a_category_does_not_admit_a_user_tool_that_shares_a_name(self) -> None:
        # `dr_db` expands to the NAMES of DataRobot's own database tools. A user who
        # names their tool `vdb_query` must not thereby land in Databases — the gallery
        # files it under dr_user_tools, and `?category=dr_db` returns nothing, so the
        # header has to agree or the picker leads somewhere the filter denies.
        catalog = [_tool("vdb_query", "USER_TOOL"), _tool("jira_get_issue")]
        transform = seeded("dr_db")
        assert await transform.transform_tools(catalog) == []

    @pytest.mark.asyncio
    async def test_the_same_category_still_admits_the_real_built_in(self) -> None:
        # The other half: an unmarked `vdb_query` genuinely is a database tool.
        catalog = [_tool("vdb_query"), _tool("jira_get_issue")]
        transform = seeded("dr_db")
        assert [t.name for t in await transform.transform_tools(catalog)] == ["vdb_query"]

    @pytest.mark.asyncio
    async def test_naming_the_user_tool_outright_still_admits_it(self) -> None:
        # Provenance, not identity: named explicitly, any tool is admitted.
        catalog = [_tool("vdb_query", "USER_TOOL")]
        transform = seeded("vdb_query")
        assert [t.name for t in await transform.transform_tools(catalog)] == ["vdb_query"]

    @pytest.mark.asyncio
    async def test_call_time_agrees_about_the_collision(self) -> None:
        catalog = [_tool("vdb_query", "USER_TOOL")]

        async def call_next(name: str, *, version: Any = None) -> Any:
            return next((t for t in catalog if t.name == name), None)

        assert await seeded("dr_db").get_tool("vdb_query", call_next) is None
        assert await seeded("vdb_query").get_tool("vdb_query", call_next) is not None

    @pytest.mark.asyncio
    async def test_an_unknown_token_still_admits_nothing(self) -> None:
        # The fail-closed rule is unchanged: a present header that matches no tool
        # denies everything rather than falling back to the full catalog.
        transform = seeded("dr_typo_nonsense")
        assert await transform.transform_tools(CATALOG) == []
