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

"""The tool-category tree behind ``GET /toolGallery/categories/``.

Builds the two-level taxonomy (parents → leaf children, plus standalone leaves and
the marker-resolved buckets) with per-node **live tool counts scoped to the tools
*this* server actually exposes**, so a filter panel can render ``Data connectors
(18)`` and a picker can show the tools inside each leaf.

Each node is ``{value, label, count, dynamic, appliesTo, toolNames, children}``.
``value`` is the ``dr_*`` string a tool item reports in its ``categories`` and the
gallery's ``?category=`` filter accepts, so a selection round-trips against
``GET /toolGallery/tools/`` without translation. ``label`` comes from
``TOOL_CATEGORY_LABELS`` — one map, used by every consumer.

``appliesTo`` names the server types a category is meaningful for: the static
taxonomy applies to ``["global", "user"]``; the marker-resolved buckets
(``dr_user_tools``, ``dr_dynamic_tools``) apply to ``["user"]`` only, since
global-mcp serves built-in tools exclusively.

Counts come from the server's real catalog (see ``resolve_catalog``: the caller's
session headers must not narrow it) mapped through the single-source-of-truth
taxonomy (``merge_tool_info`` → ``categories_for_tool``) plus the marker classifier
(``marked_kind``) — never from the static ``LEAF_CATEGORY_TOOLS`` map alone, which
would over-count tools a given server does not register. Proxied user-MCP tools
carry no category and are not bucketed anywhere.

This module has no route of its own: it used to be ``routes/tool_categories.py``
serving a separate ``GET /toolCategories/``, which meant two endpoints answered
"what categories exist" with different shapes and different completeness. The tree
IS the category endpoint now; only the builder lives here.
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from datarobot_genai.drmcputils.categories import PARENT_TO_CHILDREN
from datarobot_genai.drmcputils.categories import MCPToolCategory
from datarobot_genai.drmcputils.categories import category_label
from datarobot_genai.drmcputils.tool_gallery import marked_kind
from datarobot_genai.drmcputils.tool_gallery import merge_tool_info

# Enum definition order → stable ordering for both top-level categories and children.
_ENUM_ORDER: dict[str, int] = {category.value: i for i, category in enumerate(MCPToolCategory)}

# Parents own leaf children; ``_CHILDREN`` is every leaf that appears under a parent.
_PARENTS: frozenset[str] = frozenset(str(parent) for parent in PARENT_TO_CHILDREN)
_CHILDREN: frozenset[str] = frozenset(
    str(child) for children in PARENT_TO_CHILDREN.values() for child in children
)

# Categories resolved from tool markers at request time — no static tool names.
# Flagged ``dynamic`` so a UI can render them; counts still come from live tools.
_DYNAMIC_CATEGORIES: frozenset[str] = frozenset(
    {MCPToolCategory.DR_USER_TOOLS.value, MCPToolCategory.DR_DYNAMIC_TOOLS.value}
)

# ``appliesTo`` values per node: the marker-resolved buckets only exist on user
# MCPs (user-authored / dynamically registered tools); everything else is
# meaningful for both server types.
_APPLIES_TO_ALL = ["global", "user"]
_APPLIES_TO_USER_ONLY = ["user"]


def build_category_tree(tools: Sequence[Any]) -> tuple[list[dict[str, Any]], int]:
    """Build the category tree for *tools*.

    Returns the top-level nodes and the number of DISTINCT tools that landed in at
    least one category — which is smaller than ``len(tools)`` whenever a server
    exposes uncategorized tools (proxied user-MCP tools, foremost).
    """
    by_category = _category_to_tools(tools)
    nodes = [_category_node(name, by_category) for name in ordered_top_level()]
    mapped: set[str] = set()
    for names in by_category.values():
        mapped |= names
    return nodes, len(mapped)


def ordered_top_level() -> list[str]:
    """Top-level categories: parents first, then standalone leaves, dynamic last.

    Derived from the taxonomy (not hard-coded) so a new category flows through to
    the filter panel by existing; within each group, enum-definition order is
    preserved. This mirrors the picker layout.
    """
    tops = [c.value for c in MCPToolCategory if c.value in _PARENTS or c.value not in _CHILDREN]

    def sort_key(name: str) -> tuple[bool, bool, int]:
        # (dynamic last, non-parents after parents, then enum order)
        return (name in _DYNAMIC_CATEGORIES, name not in _PARENTS, _ENUM_ORDER[name])

    return sorted(tops, key=sort_key)


def _category_to_tools(tools: Sequence[Any]) -> dict[str, set[str]]:
    """Map each category → the set of *this server's* tool names in it.

    Static tools contribute their leaf + parent categories (``categories_for_tool``
    via ``merge_tool_info``); marker-classified tools (user/dynamic) contribute
    their bucket (``marked_kind``). A tool with no known category contributes
    nothing.
    """
    mapping: dict[str, set[str]] = defaultdict(set)
    for tool in tools:
        info = merge_tool_info(tool, {})
        names = {str(category) for category in info["categories"]}
        kind = marked_kind(info.get("tool_category"))
        # Proxied tools have a marker kind but no category → contribute nothing.
        if kind and kind["category"]:
            names.add(kind["category"])
        for name in names:
            mapping[name].add(tool.name)
    return mapping


def _ordered_children(parent: str) -> list[str]:
    """Leaf children of *parent* in enum-definition order ([] for a leaf)."""
    children = [str(child) for child in PARENT_TO_CHILDREN.get(parent, frozenset())]
    return sorted(children, key=lambda name: _ENUM_ORDER[name])


def _category_node(name: str, by_category: dict[str, set[str]]) -> dict[str, Any]:
    """Build one category node (recursing one level into leaf children)."""
    children = [_category_node(child, by_category) for child in _ordered_children(name)]
    tool_names = sorted(by_category.get(name, set()))
    dynamic = name in _DYNAMIC_CATEGORIES
    return {
        "value": name,
        "label": category_label(name),
        "count": len(tool_names),
        "dynamic": dynamic,
        "appliesTo": _APPLIES_TO_USER_ONLY if dynamic else _APPLIES_TO_ALL,
        "toolNames": tool_names,
        "children": children,
    }
