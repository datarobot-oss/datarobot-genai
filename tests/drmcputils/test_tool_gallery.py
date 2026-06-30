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

"""Tests for drmcputils.tool_gallery — gallery response builder and private key set."""

from datarobot_genai.drmcputils.tool_gallery import DRTOOLS_PRIVATE_METADATA_KEYS
from datarobot_genai.drmcputils.tool_gallery import build_tool_gallery_items


class TestDrtoolsPrivateMetadataKeys:
    def test_is_frozenset(self) -> None:
        assert isinstance(DRTOOLS_PRIVATE_METADATA_KEYS, frozenset)

    def test_contains_all_expected_keys(self) -> None:
        assert DRTOOLS_PRIVATE_METADATA_KEYS == frozenset(
            {
                "excluded_from_global_mcp",
                "display_name",
                "description_ui",
                "auth_provider",
                "categories",
            }
        )

    def test_excluded_from_global_mcp_in_set(self) -> None:
        assert "excluded_from_global_mcp" in DRTOOLS_PRIVATE_METADATA_KEYS

    def test_gallery_display_fields_in_set(self) -> None:
        for key in ("display_name", "description_ui", "auth_provider", "categories"):
            assert key in DRTOOLS_PRIVATE_METADATA_KEYS, f"{key!r} missing from private keys"


class TestBuildToolGalleryItems:
    def test_empty_list_returns_empty(self) -> None:
        assert build_tool_gallery_items([]) == []

    def test_minimal_tool_uses_safe_defaults(self) -> None:
        result = build_tool_gallery_items([{"name": "my_tool"}])
        assert len(result) == 1
        item = result[0]
        assert item["name"] == "my_tool"
        assert item["display_name"] == "my_tool"
        assert item["description_ui"] == ""
        assert item["tags"] == []
        assert item["categories"] == []
        assert item["auth_provider"] is None
        assert item["hosted"] is False

    def test_fully_populated_tool_round_trips(self) -> None:
        tool = {
            "name": "jira_search_issues",
            "display_name": "Jira — Search Issues",
            "description_ui": "Find Jira issues matching a JQL query.",
            "tags": ["jira", "search"],
            "categories": ["dr_connectors", "dr_connector_jira"],
            "auth_provider": "jira",
            "hosted": False,
        }
        result = build_tool_gallery_items([tool])
        item = result[0]
        assert item["name"] == "jira_search_issues"
        assert item["display_name"] == "Jira — Search Issues"
        assert item["description_ui"] == "Find Jira issues matching a JQL query."
        assert item["auth_provider"] == "jira"
        assert item["hosted"] is False

    def test_multiple_tools_preserved_in_order(self) -> None:
        tools = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        result = build_tool_gallery_items(tools)
        assert [r["name"] for r in result] == ["a", "b", "c"]

    # ── display_name fallback ────────────────────────────────────────────────

    def test_display_name_absent_falls_back_to_name(self) -> None:
        result = build_tool_gallery_items([{"name": "raw_name"}])
        assert result[0]["display_name"] == "raw_name"

    def test_display_name_none_falls_back_to_name(self) -> None:
        result = build_tool_gallery_items([{"name": "raw_name", "display_name": None}])
        assert result[0]["display_name"] == "raw_name"

    def test_display_name_empty_string_falls_back_to_name(self) -> None:
        result = build_tool_gallery_items([{"name": "raw_name", "display_name": ""}])
        assert result[0]["display_name"] == "raw_name"

    def test_display_name_present_takes_precedence(self) -> None:
        result = build_tool_gallery_items(
            [{"name": "raw_name", "display_name": "Human Friendly Name"}]
        )
        assert result[0]["display_name"] == "Human Friendly Name"

    # ── description_ui fallback ──────────────────────────────────────────────

    def test_description_ui_absent_becomes_empty_string(self) -> None:
        result = build_tool_gallery_items([{"name": "t"}])
        assert result[0]["description_ui"] == ""

    def test_description_ui_none_becomes_empty_string(self) -> None:
        result = build_tool_gallery_items([{"name": "t", "description_ui": None}])
        assert result[0]["description_ui"] == ""

    # ── tags sorting ─────────────────────────────────────────────────────────

    def test_tags_are_sorted_alphabetically(self) -> None:
        result = build_tool_gallery_items([{"name": "t", "tags": ["zzz", "aaa", "mmm"]}])
        assert result[0]["tags"] == ["aaa", "mmm", "zzz"]

    def test_tags_set_input_is_sorted(self) -> None:
        result = build_tool_gallery_items([{"name": "t", "tags": {"beta", "alpha"}}])
        assert result[0]["tags"] == ["alpha", "beta"]

    def test_tags_absent_becomes_empty_list(self) -> None:
        result = build_tool_gallery_items([{"name": "t"}])
        assert result[0]["tags"] == []

    def test_tags_none_becomes_empty_list(self) -> None:
        result = build_tool_gallery_items([{"name": "t", "tags": None}])
        assert result[0]["tags"] == []

    # ── categories ───────────────────────────────────────────────────────────

    def test_categories_list_preserved(self) -> None:
        cats = ["dr_connectors", "dr_connector_jira"]
        result = build_tool_gallery_items([{"name": "t", "categories": cats}])
        assert result[0]["categories"] == cats

    def test_categories_frozenset_converted_to_list(self) -> None:
        result = build_tool_gallery_items(
            [{"name": "t", "categories": frozenset({"dr_connectors"})}]
        )
        assert isinstance(result[0]["categories"], list)
        assert result[0]["categories"] == ["dr_connectors"]

    def test_categories_absent_becomes_empty_list(self) -> None:
        result = build_tool_gallery_items([{"name": "t"}])
        assert result[0]["categories"] == []

    def test_categories_none_becomes_empty_list(self) -> None:
        result = build_tool_gallery_items([{"name": "t", "categories": None}])
        assert result[0]["categories"] == []

    # ── auth_provider fallback ───────────────────────────────────────────────

    def test_auth_provider_none_stays_none(self) -> None:
        result = build_tool_gallery_items([{"name": "t", "auth_provider": None}])
        assert result[0]["auth_provider"] is None

    def test_auth_provider_empty_string_becomes_none(self) -> None:
        result = build_tool_gallery_items([{"name": "t", "auth_provider": ""}])
        assert result[0]["auth_provider"] is None

    def test_auth_provider_present_is_preserved(self) -> None:
        result = build_tool_gallery_items([{"name": "t", "auth_provider": "jira"}])
        assert result[0]["auth_provider"] == "jira"

    # ── hosted coercion ──────────────────────────────────────────────────────

    def test_hosted_false_by_default(self) -> None:
        result = build_tool_gallery_items([{"name": "t"}])
        assert result[0]["hosted"] is False

    def test_hosted_true_preserved_as_bool(self) -> None:
        result = build_tool_gallery_items([{"name": "t", "hosted": True}])
        assert result[0]["hosted"] is True

    def test_hosted_truthy_int_coerced_to_true(self) -> None:
        result = build_tool_gallery_items([{"name": "t", "hosted": 1}])
        assert result[0]["hosted"] is True

    def test_hosted_false_preserved_as_bool(self) -> None:
        result = build_tool_gallery_items([{"name": "t", "hosted": False}])
        assert result[0]["hosted"] is False
