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

"""Tests for ``register_toolsets_allowlist_expander``."""

from collections.abc import Iterator

import pytest

from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCP_TOOLSETS_HEADER
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCPRequestContext
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import effective_tool_allowlist
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import expand_toolset_names_to_tools
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import register_toolsets_allowlist_expander


@pytest.fixture(autouse=True)
def _reset_toolsets_expander() -> Iterator[None]:
    register_toolsets_allowlist_expander(None)
    yield
    register_toolsets_allowlist_expander(None)


class TestRegisterToolsetsAllowlistExpander:
    @pytest.mark.asyncio
    async def test_default_returns_empty(self) -> None:
        assert await expand_toolset_names_to_tools(frozenset({"pack"})) == frozenset()

    @pytest.mark.asyncio
    async def test_registered_expander_is_used(self) -> None:
        async def expander(names: frozenset[str]) -> frozenset[str]:
            return frozenset({"tool_a"}) if "pack" in names else frozenset()

        register_toolsets_allowlist_expander(expander)
        assert await expand_toolset_names_to_tools(frozenset({"pack"})) == frozenset({"tool_a"})

    @pytest.mark.asyncio
    async def test_expander_failure_fails_closed(self) -> None:
        async def boom(_names: frozenset[str]) -> frozenset[str]:
            raise RuntimeError("mongo down")

        register_toolsets_allowlist_expander(boom)
        assert await expand_toolset_names_to_tools(frozenset({"pack"})) == frozenset()

    @pytest.mark.asyncio
    async def test_from_headers_unions_tools_and_expanded_toolsets(self) -> None:
        register_toolsets_allowlist_expander(lambda _n: _return(frozenset({"from_toolset"})))

        ctx = MCPRequestContext.from_headers(
            {"x-datarobot-mcp-tools": "jira_search_issues", MCP_TOOLSETS_HEADER: "pack"}
        )
        allowlist = await effective_tool_allowlist(ctx)
        assert allowlist is not None
        assert "jira_search_issues" in allowlist
        assert "from_toolset" in allowlist


class TestEffectiveToolAllowlistFailsClosed:
    """An unresolvable toolsets header must narrow the request, never widen it.

    ``None`` and ``frozenset()`` are opposite answers downstream — the first disables
    allowlisting entirely, the second permits nothing — so the difference between them is
    the difference between a session seeing one bundle and a session seeing everything.
    """

    @pytest.mark.asyncio
    async def test_unresolvable_toolsets_header_alone_permits_nothing(self) -> None:
        """GIVEN only a toolsets header that expands to nothing, THEN no tool is allowed."""
        register_toolsets_allowlist_expander(lambda _n: _return(frozenset()))

        ctx = MCPRequestContext.from_headers({MCP_TOOLSETS_HEADER: "deleted-pack"})

        assert await effective_tool_allowlist(ctx) == frozenset()

    @pytest.mark.asyncio
    async def test_expander_outage_permits_nothing(self) -> None:
        """GIVEN the expander raises (Mongo down), THEN no tool is allowed."""

        async def boom(_names: frozenset[str]) -> frozenset[str]:
            raise RuntimeError("mongo down")

        register_toolsets_allowlist_expander(boom)
        ctx = MCPRequestContext.from_headers({MCP_TOOLSETS_HEADER: "pack"})

        assert await effective_tool_allowlist(ctx) == frozenset()

    @pytest.mark.asyncio
    async def test_unregistered_expander_permits_nothing(self) -> None:
        """GIVEN a server with no expander (user-mcp), THEN the header still narrows."""
        ctx = MCPRequestContext.from_headers({MCP_TOOLSETS_HEADER: "pack"})

        assert await effective_tool_allowlist(ctx) == frozenset()

    @pytest.mark.asyncio
    async def test_unresolvable_toolsets_leaves_an_explicit_tools_header_intact(self) -> None:
        """GIVEN both headers and an empty expansion, THEN only the named tools survive."""
        register_toolsets_allowlist_expander(lambda _n: _return(frozenset()))

        ctx = MCPRequestContext.from_headers(
            {"x-datarobot-mcp-tools": "jira_search_issues", MCP_TOOLSETS_HEADER: "deleted-pack"}
        )

        assert await effective_tool_allowlist(ctx) == frozenset({"jira_search_issues"})

    @pytest.mark.asyncio
    async def test_no_toolsets_header_leaves_the_allowlist_untouched(self) -> None:
        """GIVEN no toolsets header, THEN the result is exactly the tools header (or None)."""
        assert await effective_tool_allowlist(MCPRequestContext.from_headers({})) is None
        assert await effective_tool_allowlist(
            MCPRequestContext.from_headers({"x-datarobot-mcp-tools": "jira_search_issues"})
        ) == frozenset({"jira_search_issues"})


class TestEffectiveToolAllowlistCaching:
    @pytest.mark.asyncio
    async def test_expander_runs_once_per_request_context(self) -> None:
        """The expander reaches Mongo; the catalog asks for the allowlist several times."""
        calls: list[frozenset[str]] = []

        async def counting(names: frozenset[str]) -> frozenset[str]:
            calls.append(names)
            return frozenset({"tool_a"})

        register_toolsets_allowlist_expander(counting)
        ctx = MCPRequestContext.from_headers({MCP_TOOLSETS_HEADER: "pack"})

        assert await effective_tool_allowlist(ctx) == frozenset({"tool_a"})
        assert await effective_tool_allowlist(ctx) == frozenset({"tool_a"})
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_a_different_context_is_not_served_the_cached_expansion(self) -> None:
        async def per_name(names: frozenset[str]) -> frozenset[str]:
            return frozenset(f"tool_from_{name}" for name in names)

        register_toolsets_allowlist_expander(per_name)

        first = MCPRequestContext.from_headers({MCP_TOOLSETS_HEADER: "pack_a"})
        second = MCPRequestContext.from_headers({MCP_TOOLSETS_HEADER: "pack_b"})

        assert await effective_tool_allowlist(first) == frozenset({"tool_from_pack_a"})
        assert await effective_tool_allowlist(second) == frozenset({"tool_from_pack_b"})


async def _return(value: frozenset[str]) -> frozenset[str]:
    return value
