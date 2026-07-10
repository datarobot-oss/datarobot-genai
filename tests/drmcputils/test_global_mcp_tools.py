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

"""Tests for the global-mcp single source of truth (drmcputils.global_mcp_tools)."""

from datarobot_genai.drmcputils.categories import LEAF_CATEGORY_TOOLS
from datarobot_genai.drmcputils.categories import MCPToolCategory
from datarobot_genai.drmcputils.global_mcp_tools import GLOBAL_MCP_EXCLUDED_TOOLS
from datarobot_genai.drmcputils.global_mcp_tools import GLOBAL_MCP_PACKAGE_CATEGORIES
from datarobot_genai.drmcputils.global_mcp_tools import global_mcp_enabled_packages
from datarobot_genai.drmcputils.global_mcp_tools import global_mcp_leaf_categories


class TestEnabledPackages:
    def test_enabled_packages_are_the_mapping_keys(self) -> None:
        assert global_mcp_enabled_packages() == frozenset(GLOBAL_MCP_PACKAGE_CATEGORIES)

    def test_expected_packages(self) -> None:
        assert global_mcp_enabled_packages() == {
            "predictive",
            "use_case",
            "perplexity",
            "tavily",
            "dr_docs",
            "vdb",
        }


class TestLeafCategories:
    def test_is_union_of_mapping_values(self) -> None:
        expected = frozenset().union(*GLOBAL_MCP_PACKAGE_CATEGORIES.values())
        assert global_mcp_leaf_categories() == expected

    def test_every_category_is_a_known_leaf(self) -> None:
        # Guards against a typo'd / non-leaf category in the source of truth.
        for category in global_mcp_leaf_categories():
            assert isinstance(category, MCPToolCategory)
            assert category in LEAF_CATEGORY_TOOLS

    def test_categories_are_non_empty(self) -> None:
        # Each enabled leaf category actually maps to tools in the taxonomy.
        for category in global_mcp_leaf_categories():
            assert LEAF_CATEGORY_TOOLS[category], f"{category} has no tools"


class TestExcludedTools:
    def test_contains_file_upload(self) -> None:
        # file_upload needs local disk access → blocked from global-mcp here (the single
        # source of truth for exclusions), even though files_api isn't enabled yet.
        assert "file_upload" in GLOBAL_MCP_EXCLUDED_TOOLS

    def test_excluded_tools_never_advertised_by_ard(self) -> None:
        # Excluded names must not leak into the ARD catalog.
        from datarobot_genai.drmcputils.ard_catalog import get_global_mcp_prebuilt_tools

        names = {t["name"] for t in get_global_mcp_prebuilt_tools()["tools"]}
        assert names.isdisjoint(GLOBAL_MCP_EXCLUDED_TOOLS)
