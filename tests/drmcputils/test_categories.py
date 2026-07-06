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


import importlib
import importlib.util
import pkgutil

from datarobot_genai.drmcputils.categories import LEAF_CATEGORY_TOOLS
from datarobot_genai.drmcputils.categories import PARENT_TO_CHILDREN
from datarobot_genai.drmcputils.categories import MCPToolCategory
from datarobot_genai.drmcputils.categories import categories_for_tool
from datarobot_genai.drmcputils.categories import parse_tool_allowlist_header
from datarobot_genai.drmcputils.categories import resolve_to_tool_names
from datarobot_genai.drtools.core import get_registered_tools


def _all_live_drtools_tool_names() -> set[str]:
    """Import every drtools tool module and return the registered tool names.

    Mirrors the package walk both registry loaders perform: recurse through
    ``datarobot_genai.drtools`` (skipping ``core`` and private modules) so the
    ``@tool_metadata`` decorators populate the registry, then read it back.
    Commented-out decorators never execute, so they are correctly excluded.
    """

    def _walk(package_name: str) -> None:
        spec = importlib.util.find_spec(package_name)
        if spec is None or not spec.submodule_search_locations:
            return
        for info in pkgutil.iter_modules(spec.submodule_search_locations, package_name + "."):
            last = info.name.rsplit(".", 1)[-1]
            if last.startswith("_"):
                continue
            if package_name == "datarobot_genai.drtools" and last == "core":
                continue
            if info.ispkg:
                _walk(info.name)
            else:
                importlib.import_module(info.name)

    _walk("datarobot_genai.drtools")
    return {(metadata.get("name") or func.__name__) for func, metadata in get_registered_tools()}


class TestResolveToToolNames:
    def test_parent_category_expands_to_all_leaf_tools(self):
        result = resolve_to_tool_names(frozenset({"dr_connectors"}))
        # Must contain all tools from all four connector sub-categories
        assert "jira_search_issues" in result
        assert "confluence_get_page" in result
        assert "gdrive_find_contents" in result
        assert "microsoft_graph_search_content" in result

    def test_leaf_category_expands_to_its_tools(self):
        result = resolve_to_tool_names(frozenset({"dr_connector_jira"}))
        assert result == LEAF_CATEGORY_TOOLS[MCPToolCategory.DR_CONNECTOR_JIRA]

    def test_plain_tool_name_passes_through(self):
        result = resolve_to_tool_names(frozenset({"jira_search_issues"}))
        assert "jira_search_issues" in result

    def test_unknown_entry_passes_through(self):
        result = resolve_to_tool_names(frozenset({"dr_typo_category"}))
        # Unknown entries are kept as-is — they will simply not match any tool
        assert "dr_typo_category" in result

    def test_mix_of_category_and_plain_tool(self):
        result = resolve_to_tool_names(frozenset({"dr_web_search_tavily", "jira_get_issue"}))
        assert "tavily_search_web" in result
        assert "tavily_extract_text" in result
        assert "jira_get_issue" in result
        # Jira tools should NOT be included (only tavily was requested)
        assert "jira_search_issues" not in result

    def test_empty_frozenset_returns_empty(self):
        assert resolve_to_tool_names(frozenset()) == frozenset()

    def test_parent_dr_predictive_contains_all_children(self):
        result = resolve_to_tool_names(frozenset({"dr_predictive"}))
        # Spot-check tools from each child category
        assert "catalog_list_datasets" in result  # dr_catalog
        assert "modeling_list_projects" in result  # dr_modeling
        assert "predict_get_batch_results" in result  # dr_predictions
        assert "datarobot_usecases_list" not in result
        assert "deployment_get_list" not in result

    def test_parent_children_are_leaf_categories(self):
        for parent, children in PARENT_TO_CHILDREN.items():
            for child in children:
                assert child in LEAF_CATEGORY_TOOLS, (
                    f"Child {child!r} of parent {parent!r} not found in LEAF_CATEGORY_TOOLS"
                )

    def test_all_leaf_categories_have_tool_sets(self):
        """Every leaf category must map to a frozenset (empty is allowed for placeholders)."""
        for cat, tools in LEAF_CATEGORY_TOOLS.items():
            assert isinstance(tools, frozenset), f"{cat} mapped to {type(tools)}"

    def test_hosted_categories_expand_to_empty_and_pass_through(self):
        # dr_proxied_user_mcp and dr_dynamic_tools map to empty frozensets;
        # entries are recognised as categories, so nothing is passed through.
        result = resolve_to_tool_names(frozenset({"dr_proxied_user_mcp"}))
        assert result == frozenset()


class TestResolveToToolNamesAdditional:
    """Additional coverage for resolve_to_tool_names across all parent categories."""

    def test_dr_web_search_expands_to_perplexity_and_tavily(self) -> None:
        result = resolve_to_tool_names(frozenset({"dr_web_search"}))
        assert "perplexity_search" in result
        assert "perplexity_sonar" in result
        assert "tavily_search_web" in result
        assert "tavily_extract_text" in result

    def test_dr_documentation_is_a_leaf_not_a_parent(self) -> None:
        # dr_documentation has no sub-categories — it should expand directly to tools.
        result = resolve_to_tool_names(frozenset({"dr_documentation"}))
        assert "search_datarobot_agentic_docs" in result
        assert "datarobot_docs_fetch_page" in result

    def test_dr_db_parent_expands_to_vdb_tools(self) -> None:
        result = resolve_to_tool_names(frozenset({"dr_db"}))
        assert "vdb_list" in result
        assert "vdb_query" in result

    def test_dr_development_expands_to_workload_and_file_tools(self) -> None:
        result = resolve_to_tool_names(frozenset({"dr_development"}))
        assert "workload_list" in result
        assert "workload_create" in result
        assert "file_import" in result
        assert "file_read" in result

    def test_dr_visual_expands_to_panels_tools(self) -> None:
        result = resolve_to_tool_names(frozenset({"dr_visual"}))
        assert "list_panels" in result
        assert "get_panel" in result
        assert "transform_panel" in result

    def test_dr_dynamic_tools_maps_to_empty_frozenset(self) -> None:
        result = resolve_to_tool_names(frozenset({"dr_dynamic_tools"}))
        assert result == frozenset()

    def test_multiple_parents_in_one_call_union_results(self) -> None:
        result = resolve_to_tool_names(frozenset({"dr_web_search", "dr_documentation"}))
        assert "tavily_search_web" in result
        assert "search_datarobot_agentic_docs" in result

    def test_file_upload_present_in_dr_file(self) -> None:
        # file_upload is in the category definition; exclusion happens at registration time
        # (via excluded_from_global_mcp flag), not in the category map.
        assert "file_upload" in LEAF_CATEGORY_TOOLS[MCPToolCategory.DR_FILE]

    def test_no_tool_name_appears_in_multiple_leaf_categories(self) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for cat, tools in LEAF_CATEGORY_TOOLS.items():
            for tool in tools:
                if tool in seen:
                    duplicates.append(tool)
                seen.add(tool)
        assert duplicates == [], f"Tool names duplicated across categories: {duplicates}"

    def test_all_parent_to_children_values_are_leaf_categories(self) -> None:
        for parent, children in PARENT_TO_CHILDREN.items():
            for child in children:
                assert child in LEAF_CATEGORY_TOOLS, (
                    f"Child {child!r} of {parent!r} missing from LEAF_CATEGORY_TOOLS"
                )

    def test_all_mcp_tool_category_members_are_string_enum(self) -> None:
        for member in MCPToolCategory:
            assert isinstance(member, str), f"{member!r} is not a str"
            assert member == member.value  # StrEnum: member == its string value


class TestParseToolAllowlistHeader:
    def test_none_header_returns_none(self) -> None:
        assert parse_tool_allowlist_header(None) is None

    def test_blank_header_returns_none(self) -> None:
        assert parse_tool_allowlist_header("   ") is None

    def test_only_commas_returns_none(self) -> None:
        assert parse_tool_allowlist_header(",,,") is None

    def test_category_name_is_resolved(self) -> None:
        result = parse_tool_allowlist_header("dr_connector_jira")
        assert result is not None
        assert "jira_search_issues" in result

    def test_plain_tool_name_is_kept(self) -> None:
        result = parse_tool_allowlist_header("jira_search_issues,confluence_get_page")
        assert result is not None
        assert "jira_search_issues" in result
        assert "confluence_get_page" in result

    def test_whitespace_around_entries_ignored(self) -> None:
        result = parse_tool_allowlist_header("  dr_connector_jira , jira_get_issue  ")
        assert result is not None
        assert "jira_search_issues" in result
        assert "jira_get_issue" in result

    def test_typo_category_kept_as_plain_name(self) -> None:
        result = parse_tool_allowlist_header("dr_typo_xyz")
        assert result is not None
        assert "dr_typo_xyz" in result


class TestCategoriesForTool:
    """Reverse index: tool name → its leaf category plus parent (if any)."""

    def test_tool_under_parented_leaf_returns_leaf_and_parent(self) -> None:
        # jira_search_issues → leaf dr_connector_jira → parent dr_connectors
        assert categories_for_tool("jira_search_issues") == [
            "dr_connector_jira",
            "dr_connectors",
        ]

    def test_tool_under_standalone_leaf_returns_only_leaf(self) -> None:
        # dr_documentation is a leaf with no parent.
        assert categories_for_tool("search_datarobot_agentic_docs") == ["dr_documentation"]

    def test_use_case_tool_returns_leaf_only(self) -> None:
        assert categories_for_tool("datarobot_usecases_list") == ["dr_use_cases"]

    def test_deployment_tool_returns_leaf_only(self) -> None:
        assert categories_for_tool("deployment_get_list") == ["dr_deployments"]

    def test_predictive_tool_returns_leaf_and_predictive_parent(self) -> None:
        assert categories_for_tool("modeling_list_models") == ["dr_modeling", "dr_predictive"]

    def test_result_is_sorted(self) -> None:
        result = categories_for_tool("jira_search_issues")
        assert result == sorted(result)

    def test_unknown_tool_returns_empty_list(self) -> None:
        assert categories_for_tool("not_a_real_tool") == []

    def test_hosted_category_tool_names_are_not_indexed(self) -> None:
        # dr_proxied_user_mcp / dr_dynamic_tools map to empty tool sets, so no
        # tool names are indexed under them.
        assert categories_for_tool("dr_proxied_user_mcp") == []

    def test_every_categorized_tool_round_trips(self) -> None:
        # Each tool in every leaf category must report that leaf among its categories.
        for leaf, tools in LEAF_CATEGORY_TOOLS.items():
            for tool_name in tools:
                assert leaf in categories_for_tool(tool_name)


class TestTaxonomyCompleteness:
    """Guard against drift between live @tool_metadata tools and the taxonomy.

    A tool that exists but is in no leaf category is silently dropped whenever a
    request filters by that tool's category — the failure mode is invisible at
    runtime, so it must be caught here.
    """

    def test_every_live_tool_belongs_to_a_leaf_category(self) -> None:
        live = _all_live_drtools_tool_names()
        categorized: set[str] = set().union(*LEAF_CATEGORY_TOOLS.values())
        uncategorized = live - categorized
        assert not uncategorized, (
            "These live @tool_metadata tools are missing from LEAF_CATEGORY_TOOLS "
            f"and will be dropped by category filters: {sorted(uncategorized)}"
        )

    def test_no_stale_tool_names_in_taxonomy(self) -> None:
        live = _all_live_drtools_tool_names()
        categorized: set[str] = set().union(*LEAF_CATEGORY_TOOLS.values())
        stale = categorized - live
        assert not stale, (
            "These tool names appear in LEAF_CATEGORY_TOOLS but no longer have a "
            f"live @tool_metadata decorator: {sorted(stale)}"
        )
