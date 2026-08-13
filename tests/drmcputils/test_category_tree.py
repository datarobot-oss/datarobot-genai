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

"""The tree behind ``GET /toolGallery/categories/`` must agree with ``/toolGallery/tools/``."""

from types import SimpleNamespace
from typing import Any

from datarobot_genai.drmcputils.category_tree import build_category_tree
from datarobot_genai.drmcputils.tool_gallery import build_tool_gallery_items
from datarobot_genai.drmcputils.tool_gallery import merge_tool_info


def _tool(name: str, tool_category: str | None = None) -> SimpleNamespace:
    """Build a stand-in for a FastMCP ``Tool`` — the attrs the builders actually read."""
    meta = {"tool_category": tool_category} if tool_category else {}
    return SimpleNamespace(name=name, description=f"{name} description", tags=set(), meta=meta)


def _populated_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every node with at least one tool, parents and children alike."""
    found = []
    for node in nodes:
        if node["count"]:
            found.append(node)
        found.extend(_populated_nodes(node["children"]))
    return found


def _filterable(tools: list[Any]) -> dict[str, set[str]]:
    """Category value → tool names ``GET /toolGallery/tools/?category=<value>`` returns."""
    mapping: dict[str, set[str]] = {}
    for item in build_tool_gallery_items([merge_tool_info(t, {}) for t in tools]):
        for category in item["categories"]:
            mapping.setdefault(str(category), set()).add(item["name"])
    return mapping


class TestTreeAgreesWithTheToolsFilter:
    """A category the tree shows N tools in must return those N tools when filtered on.

    These are the same endpoint pair from the caller's point of view — the picker is
    built from one and clicked through to the other — so any divergence is a dead end
    in the UI, not a cosmetic mismatch.
    """

    def test_marked_tool_is_bucketed_only_where_the_filter_can_find_it(self) -> None:
        # GIVEN a user-authored tool whose name collides with the static taxonomy
        # (``file_read`` is a real drtools tool name under dr_development/dr_file)
        tools = [_tool("file_read", tool_category="USER_TOOL")]

        # WHEN the tree is built
        nodes, _ = build_category_tree(tools)

        # THEN it is filed under its marker bucket alone. Unioning the marker bucket with
        # the static taxonomy filed it under Development -> Files too, where clicking
        # through returned nothing: the gallery treats the two as mutually exclusive.
        assert {node["value"] for node in _populated_nodes(nodes)} == {"dr_user_tools"}

    def test_every_populated_node_round_trips_through_the_filter(self) -> None:
        # GIVEN a server mixing static, user, dynamic and proxied tools
        tools = [
            _tool("jira_search_issues"),
            _tool("file_read", tool_category="USER_TOOL"),
            _tool("my_own_tool", tool_category="USER_TOOL"),
            _tool("some_deployment", tool_category="USER_TOOL_DEPLOYMENT"),
            _tool("proxied_thing", tool_category="PROXIED_USER_MCP"),
        ]
        nodes, _ = build_category_tree(tools)
        filterable = _filterable(tools)

        # THEN each node's tools are exactly what ?category=<value> would return
        for node in _populated_nodes(nodes):
            assert set(node["toolNames"]) == filterable.get(node["value"], set()), (
                f"node {node['value']} offers tools the ?category= filter does not return"
            )

    def test_static_tool_counts_under_both_its_leaf_and_its_parent(self) -> None:
        # A parent has to count everything beneath it or "Data connectors (0)" would sit
        # above a populated "Jira".
        nodes, total = build_category_tree([_tool("jira_search_issues")])
        populated = {node["value"]: node["toolNames"] for node in _populated_nodes(nodes)}
        assert populated == {
            "dr_connectors": ["jira_search_issues"],
            "dr_connector_jira": ["jira_search_issues"],
        }
        # ...but the tool is still ONE tool: totalCount is distinct tools, not memberships.
        assert total == 1

    def test_proxied_tools_are_counted_nowhere(self) -> None:
        # Proxied user-MCP tools carry no category (the dr_proxied_user_mcp bucket was
        # removed), so they belong to no node and do not inflate totalCount.
        nodes, total = build_category_tree(
            [_tool("proxied_thing", tool_category="PROXIED_USER_MCP")]
        )
        assert _populated_nodes(nodes) == []
        assert total == 0

    def test_a_proxied_tool_is_counted_nowhere_even_when_its_name_collides(self) -> None:
        # The marker is authoritative, so a proxied tool reports NO categories however
        # its name reads. The union this replaced left the static names in place for
        # proxied tools specifically — `kind["category"]` is None for them, so the
        # "add the bucket" branch was skipped while `categories_for_tool` had already
        # filed the tool under Databases. The gallery said `categories: []`; the tree
        # said "Databases (1)".
        nodes, total = build_category_tree([_tool("vdb_query", tool_category="PROXIED_USER_MCP")])
        assert _populated_nodes(nodes) == []
        assert total == 0
        assert _filterable([_tool("vdb_query", tool_category="PROXIED_USER_MCP")]) == {}
