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

"""Tests for the ARD catalog (drmcputils.ard_catalog)."""

import json

import pytest

from datarobot_genai.drmcputils.ard_catalog import GLOBAL_MCP_URL
from datarobot_genai.drmcputils.ard_catalog import get_global_mcp_prebuilt_tools


@pytest.fixture
def catalog() -> dict:
    return get_global_mcp_prebuilt_tools()


@pytest.fixture
def names(catalog: dict) -> set[str]:
    return {t["name"] for t in catalog["tools"]}


@pytest.fixture
def categories(catalog: dict) -> set[str]:
    return {t["category"] for t in catalog["tools"]}


class TestShape:
    def test_top_level_keys(self, catalog: dict) -> None:
        assert set(catalog.keys()) == {"mcp_url", "tools", "count"}

    def test_count_matches_tools(self, catalog: dict) -> None:
        assert catalog["count"] == len(catalog["tools"])

    def test_each_tool_has_name_and_category(self, catalog: dict) -> None:
        for tool in catalog["tools"]:
            assert set(tool.keys()) == {"name", "category"}
            assert isinstance(tool["name"], str)
            assert isinstance(tool["category"], str)

    def test_tools_sorted_by_category_then_name(self, catalog: dict) -> None:
        assert catalog["tools"] == sorted(
            catalog["tools"], key=lambda t: (t["category"], t["name"])
        )

    def test_is_json_serializable(self, catalog: dict) -> None:
        # category must be a plain string, not an enum repr.
        assert json.loads(json.dumps(catalog)) == catalog


# Leaf categories global-mcp registers — derived from the single source of truth
# (GLOBAL_MCP_PACKAGE_CATEGORIES). The global-mcp cross-check test guards this against actual
# registration; here we assert the catalog advertises exactly these and nothing from the
# not-yet-registered packages.
_ENABLED_CATEGORIES = {
    "dr_catalog",
    "dr_modeling",
    "dr_deployments",
    "dr_predictions",
    "dr_use_cases",
    "dr_web_search_perplexity",
    "dr_web_search_tavily",
    "dr_documentation",
    "dr_vdb",
    "dr_file",
    "dr_workload",
}


class TestExcludedSurface:
    """Packages global-mcp does NOT register yet must not be advertised."""

    def test_connectors_are_excluded(self, names: set[str]) -> None:
        assert not (
            names
            & {
                "jira_search_issues",
                "confluence_get_page",
                "gdrive_find_contents",
                "microsoft_graph_search_content",
            }
        )

    def test_panels_and_file_upload_excluded(self, names: set[str]) -> None:
        # Panels not registered in global-mcp yet; file_upload needs local disk access.
        assert not (names & {"file_upload", "list_panels"})

    def test_hosted_and_placeholder_categories_excluded(self, categories: set[str]) -> None:
        assert not (categories & {"dr_dynamic_tools", "dr_proxied_user_mcp", "dr_mcpapps"})

    def test_no_unexpected_categories(self, categories: set[str]) -> None:
        # Only the registered leaf categories appear — nothing leaks in.
        assert categories == _ENABLED_CATEGORIES


class TestIncludedSurface:
    def test_categories_are_exactly_the_enabled_set(self, categories: set[str]) -> None:
        assert categories == _ENABLED_CATEGORIES

    def test_predictive_web_docs_vdb_included(self, names: set[str]) -> None:
        assert {
            "modeling_list_models",
            "catalog_list_datasets",
            "deployment_get_list",
            "predict_score_catalog_realtime",
            "datarobot_usecases_list",
            "perplexity_search",
            "tavily_search_web",
            "search_datarobot_agentic_docs",
            "vdb_list",
        } <= names

    def test_files_workload_included(self, names: set[str]) -> None:
        assert {
            "file_import",
            "file_list",
            "workload_list",
            "artifact_get",
        } <= names
        assert "file_upload" not in names


class TestMcpUrl:
    def test_defaults_to_constant(self, catalog: dict) -> None:
        assert catalog["mcp_url"] == GLOBAL_MCP_URL

    def test_mcp_url_argument_overrides_default(self) -> None:
        result = get_global_mcp_prebuilt_tools(mcp_url="https://staging.example/mcp")
        assert result["mcp_url"] == "https://staging.example/mcp"
