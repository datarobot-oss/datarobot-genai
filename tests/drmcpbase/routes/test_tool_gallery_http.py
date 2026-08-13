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

"""HTTP integration tests for the shared ``GET /toolGallery/tools/`` route."""

from typing import Any

from fastmcp import FastMCP
from starlette.testclient import TestClient

from datarobot_genai.drmcputils.category_tree import ordered_top_level
from datarobot_genai.drmcputils.routes.tool_gallery import register_tool_gallery_routes


def _make_server_with_route(extra: Any = None) -> FastMCP:
    mcp = FastMCP("tool-gallery-test")

    @mcp.tool
    def jira_search_issues(a: int) -> int:
        """Search."""
        return a

    @mcp.tool
    def perplexity_search(q: str) -> str:
        """Search web."""
        return q

    register_tool_gallery_routes(mcp)
    return mcp


def _make_server_with_route_gated(gate: Any) -> FastMCP:
    mcp = FastMCP("tool-gallery-gated")

    @mcp.tool
    def jira_search_issues(a: int) -> int:
        """Search."""
        return a

    register_tool_gallery_routes(mcp, gate=gate)
    return mcp


class TestToolGalleryRoute:
    def test_returns_full_catalog_with_shape(self) -> None:
        mcp = _make_server_with_route()
        with TestClient(mcp.http_app()) as client:
            resp = client.get("/toolGallery/tools/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == len(body["tools"])
        names = {t["name"] for t in body["tools"]}
        assert {"jira_search_issues", "perplexity_search"} <= names

    def test_categories_are_derived_for_known_tools(self) -> None:
        mcp = _make_server_with_route()
        with TestClient(mcp.http_app()) as client:
            body = client.get("/toolGallery/tools/").json()
        by_name = {t["name"]: t for t in body["tools"]}
        assert by_name["jira_search_issues"]["categories"] == [
            "dr_connector_jira",
            "dr_connectors",
        ]
        assert by_name["perplexity_search"]["categories"] == [
            "dr_web_search",
            "dr_web_search_perplexity",
        ]

    def test_every_item_has_required_fields(self) -> None:
        mcp = _make_server_with_route()
        with TestClient(mcp.http_app()) as client:
            body = client.get("/toolGallery/tools/").json()
        required = {
            "name",
            "display_name",
            "description",
            "tags",
            "categories",
            "provider",
            "oauth_provider_type",
            "hosted",
        }
        for item in body["tools"]:
            assert required <= set(item.keys())

    def test_response_has_pagination_envelope(self) -> None:
        mcp = _make_server_with_route()
        with TestClient(mcp.http_app()) as client:
            body = client.get("/toolGallery/tools/").json()
        total = body["totalCount"]
        assert total == len(body["tools"])
        assert body["count"] == len(body["tools"])
        assert body["limit"] == 100
        assert body["offset"] == 0
        assert body["hasMore"] is False

    def test_limit_and_offset_paginate(self) -> None:
        mcp = _make_server_with_route()
        with TestClient(mcp.http_app()) as client:
            first = client.get("/toolGallery/tools/", params={"limit": 1, "offset": 0}).json()
            second = client.get("/toolGallery/tools/", params={"limit": 1, "offset": 1}).json()
        assert first["count"] == 1
        assert first["limit"] == 1
        assert first["totalCount"] == 2
        assert first["hasMore"] is True
        assert second["offset"] == 1
        assert second["hasMore"] is False
        # Distinct pages, no overlap.
        assert first["tools"][0]["name"] != second["tools"][0]["name"]

    def test_offset_beyond_total_returns_empty_page(self) -> None:
        mcp = _make_server_with_route()
        with TestClient(mcp.http_app()) as client:
            body = client.get("/toolGallery/tools/", params={"offset": 99}).json()
        assert body["tools"] == []
        assert body["count"] == 0
        assert body["offset"] == 99
        assert body["totalCount"] == 2
        assert body["hasMore"] is False

    def test_malformed_pagination_falls_back_to_defaults(self) -> None:
        # Non-integer query params must not 500 the gallery; they fall back to defaults.
        mcp = _make_server_with_route()
        with TestClient(mcp.http_app()) as client:
            resp = client.get("/toolGallery/tools/", params={"limit": "abc", "offset": "xyz"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["limit"] == 100
        assert body["offset"] == 0
        assert body["count"] == body["totalCount"] == 2

    def test_negative_pagination_falls_back_to_defaults(self) -> None:
        mcp = _make_server_with_route()
        with TestClient(mcp.http_app()) as client:
            body = client.get("/toolGallery/tools/", params={"limit": -5, "offset": -1}).json()
        assert body["limit"] == 100
        assert body["offset"] == 0
        assert body["count"] == 2

    def test_provider_classification(self) -> None:
        mcp = _make_server_with_route()
        with TestClient(mcp.http_app()) as client:
            body = client.get("/toolGallery/tools/").json()
        by_name = {t["name"]: t for t in body["tools"]}
        # No drtools UI metadata registered in this lightweight server, so auth_provider
        # is absent → provider defaults to datarobot, oauth_provider_type null.
        assert by_name["jira_search_issues"]["provider"] == "datarobot"
        assert by_name["jira_search_issues"]["oauth_provider_type"] is None

    def test_custom_base_path_is_honored(self) -> None:
        mcp = FastMCP("custom-path")

        @mcp.tool
        def vdb_list() -> int:
            """List."""
            return 1

        register_tool_gallery_routes(mcp, base_path="/prefixed/toolGallery")
        with TestClient(mcp.http_app()) as client:
            assert client.get("/prefixed/toolGallery/tools/").status_code == 200
            assert client.get("/toolGallery/tools/").status_code == 404


class TestToolGalleryFilters:
    """``name``, ``provider`` and ``category`` query filters, applied before pagination."""

    def _server(self) -> FastMCP:
        # jira gets an auth_provider (→ third_party); perplexity has none (→ datarobot),
        # so the two tools land in different providers for filter assertions.
        mcp = FastMCP("tool-gallery-filters")

        @mcp.tool
        def jira_search_issues(a: int) -> int:
            """Search."""
            return a

        @mcp.tool
        def perplexity_search(q: str) -> str:
            """Search web."""
            return q

        def provider() -> dict[str, dict[str, Any]]:
            return {"jira_search_issues": {"auth_provider": "jira"}}

        register_tool_gallery_routes(mcp, ui_metadata_provider=provider)
        return mcp

    def test_name_filter_returns_exact_match(self) -> None:
        with TestClient(self._server().http_app()) as client:
            body = client.get("/toolGallery/tools/", params={"name": "jira_search_issues"}).json()
        assert body["totalCount"] == 1
        assert [t["name"] for t in body["tools"]] == ["jira_search_issues"]

    def test_name_filter_unknown_returns_empty(self) -> None:
        with TestClient(self._server().http_app()) as client:
            body = client.get("/toolGallery/tools/", params={"name": "nope"}).json()
        assert body["tools"] == []
        assert body["totalCount"] == 0
        assert body["hasMore"] is False

    def test_provider_filter_third_party(self) -> None:
        with TestClient(self._server().http_app()) as client:
            body = client.get("/toolGallery/tools/", params={"provider": "third_party"}).json()
        assert [t["name"] for t in body["tools"]] == ["jira_search_issues"]
        assert body["totalCount"] == 1

    def test_provider_filter_datarobot(self) -> None:
        with TestClient(self._server().http_app()) as client:
            body = client.get("/toolGallery/tools/", params={"provider": "datarobot"}).json()
        assert [t["name"] for t in body["tools"]] == ["perplexity_search"]
        assert body["totalCount"] == 1

    def test_unknown_provider_returns_empty_page(self) -> None:
        # An unrecognised provider matches nothing rather than 500ing.
        with TestClient(self._server().http_app()) as client:
            resp = client.get("/toolGallery/tools/", params={"provider": "nope"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["tools"] == []
        assert body["totalCount"] == 0

    def test_category_filter_matches_parent_category(self) -> None:
        # dr_connectors is the parent category carried by jira_search_issues' categories.
        with TestClient(self._server().http_app()) as client:
            body = client.get("/toolGallery/tools/", params={"category": "dr_connectors"}).json()
        assert [t["name"] for t in body["tools"]] == ["jira_search_issues"]
        assert body["totalCount"] == 1

    def test_category_filter_web_search(self) -> None:
        with TestClient(self._server().http_app()) as client:
            body = client.get("/toolGallery/tools/", params={"category": "dr_web_search"}).json()
        assert [t["name"] for t in body["tools"]] == ["perplexity_search"]
        assert body["totalCount"] == 1

    def test_category_filter_valid_but_unmatched_returns_empty(self) -> None:
        # A known gallery category with no matching tool in this server yields an empty page.
        with TestClient(self._server().http_app()) as client:
            body = client.get("/toolGallery/tools/", params={"category": "dr_predictive"}).json()
        assert body["tools"] == []
        assert body["totalCount"] == 0

    def test_unknown_category_returns_empty_page(self) -> None:
        # An unrecognised category matches nothing rather than 500ing.
        with TestClient(self._server().http_app()) as client:
            resp = client.get("/toolGallery/tools/", params={"category": "dr_bogus"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["tools"] == []
        assert body["totalCount"] == 0

    def test_provider_and_category_combine(self) -> None:
        # Filters are AND-ed: third_party + dr_connectors both point at jira_search_issues.
        with TestClient(self._server().http_app()) as client:
            body = client.get(
                "/toolGallery/tools/",
                params={"provider": "third_party", "category": "dr_connectors"},
            ).json()
        assert [t["name"] for t in body["tools"]] == ["jira_search_issues"]

    def test_multiple_categories_match_any(self) -> None:
        # Repeated category params are OR-ed within the dimension (multi-select checkboxes).
        with TestClient(self._server().http_app()) as client:
            body = client.get(
                "/toolGallery/tools/",
                params={"category": ["dr_connectors", "dr_web_search"]},
            ).json()
        assert body["totalCount"] == 2
        assert {t["name"] for t in body["tools"]} == {"jira_search_issues", "perplexity_search"}

    def test_comma_separated_categories_match_any(self) -> None:
        # A single comma-separated value is equivalent to repeated params (FE join(",")).
        with TestClient(self._server().http_app()) as client:
            body = client.get(
                "/toolGallery/tools/", params={"category": "dr_connectors,dr_web_search"}
            ).json()
        assert body["totalCount"] == 2
        assert {t["name"] for t in body["tools"]} == {"jira_search_issues", "perplexity_search"}

    def test_comma_and_repeated_params_combine(self) -> None:
        # Mixing comma-separated and repeated params flattens into one match-any list.
        with TestClient(self._server().http_app()) as client:
            body = client.get(
                "/toolGallery/tools/",
                params={"category": ["dr_connectors,dr_bogus", "dr_web_search"]},
            ).json()
        assert body["totalCount"] == 2
        assert {t["name"] for t in body["tools"]} == {"jira_search_issues", "perplexity_search"}

    def test_multiple_providers_match_any(self) -> None:
        with TestClient(self._server().http_app()) as client:
            body = client.get(
                "/toolGallery/tools/",
                params={"provider": ["datarobot", "third_party"]},
            ).json()
        assert body["totalCount"] == 2
        assert {t["name"] for t in body["tools"]} == {"jira_search_issues", "perplexity_search"}

    def test_multiple_categories_ignore_unknown_values(self) -> None:
        # A mix of known and unknown categories keeps the known matches; unknown match nothing.
        with TestClient(self._server().http_app()) as client:
            body = client.get(
                "/toolGallery/tools/",
                params={"category": ["dr_connectors", "dr_bogus"]},
            ).json()
        assert [t["name"] for t in body["tools"]] == ["jira_search_issues"]

    def test_blank_filters_are_ignored(self) -> None:
        # Blank values behave like absent params — the full catalog is returned.
        with TestClient(self._server().http_app()) as client:
            body = client.get(
                "/toolGallery/tools/", params={"name": "", "provider": "", "category": ""}
            ).json()
        assert body["totalCount"] == 2

    def test_filter_totalcount_reflects_filtered_set_before_pagination(self) -> None:
        # totalCount/hasMore describe the filtered set, not the whole catalog.
        with TestClient(self._server().http_app()) as client:
            body = client.get(
                "/toolGallery/tools/", params={"provider": "third_party", "limit": 1}
            ).json()
        assert body["totalCount"] == 1
        assert body["count"] == 1
        assert body["hasMore"] is False


class TestToolGalleryCategoriesRoute:
    def test_returns_value_label_items(self) -> None:
        # GIVEN a server with the gallery routes registered
        mcp = _make_server_with_route()
        # WHEN the categories enum route is requested
        with TestClient(mcp.http_app()) as client:
            resp = client.get("/toolGallery/categories/")
        # THEN it returns 200 with {value, label} items and a matching count
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == len(body["categories"])
        assert all({"value", "label"} <= set(item.keys()) for item in body["categories"])

    def test_values_are_raw_dr_categories_with_labels(self) -> None:
        # GIVEN the gallery routes
        mcp = _make_server_with_route()
        # WHEN the categories enum is fetched
        with TestClient(mcp.http_app()) as client:
            body = client.get("/toolGallery/categories/").json()
        # THEN values are the raw dr_* strings paired with UI labels
        by_value = {item["value"]: item["label"] for item in body["categories"]}
        assert by_value["dr_connectors"] == "Data connectors"
        assert by_value["dr_web_search"] == "Web search"
        assert by_value["dr_predictive"] == "Predictive"

    def test_offers_every_top_level_category_including_marker_buckets(self) -> None:
        # GIVEN the gallery routes
        mcp = _make_server_with_route()
        # WHEN the categories route is fetched
        with TestClient(mcp.http_app()) as client:
            body = client.get("/toolGallery/categories/").json()
        values = {item["value"] for item in body["categories"]}
        # THEN every top-level category is filterable — the marker buckets included.
        # They were excluded as "internal" while this was a curated list, which left a
        # user MCP offering five categories it has no tools in and hiding the one it
        # does: dr_user_tools is most of a user MCP's catalog.
        assert values == set(ordered_top_level())
        assert {"dr_user_tools", "dr_dynamic_tools", "dr_db", "dr_deployments"} <= values

    def test_nodes_carry_live_counts_children_and_applies_to(self) -> None:
        # GIVEN a server exposing one jira tool
        mcp = _make_server_with_route()
        # WHEN the categories route is fetched
        with TestClient(mcp.http_app()) as client:
            body = client.get("/toolGallery/categories/").json()
        by_value = {item["value"]: item for item in body["categories"]}
        connectors = by_value["dr_connectors"]
        # THEN the parent counts THIS server's tools and names them, one level down
        assert connectors["count"] == 1
        assert connectors["toolNames"] == ["jira_search_issues"]
        assert connectors["appliesTo"] == ["global", "user"]
        assert connectors["dynamic"] is False
        jira = {child["value"]: child for child in connectors["children"]}["dr_connector_jira"]
        assert jira["label"] == "Jira"
        assert jira["toolNames"] == ["jira_search_issues"]
        # A category this server has nothing in still appears, at zero — a picker has
        # to be able to show what it cannot offer.
        assert by_value["dr_predictive"]["count"] == 0
        # Marker buckets are user-MCP-only and flagged dynamic.
        assert by_value["dr_user_tools"]["appliesTo"] == ["user"]
        assert by_value["dr_user_tools"]["dynamic"] is True

    def test_total_count_is_distinct_categorized_tools(self) -> None:
        # GIVEN a server with one categorized tool
        mcp = _make_server_with_route()
        # WHEN the categories route is fetched
        with TestClient(mcp.http_app()) as client:
            body = client.get("/toolGallery/categories/").json()
        # THEN count is the node count and totalCount the DISTINCT tools mapped: each
        # of the two tools sits in both a leaf and its parent, and is counted once.
        assert body["count"] == len(body["categories"])
        assert body["totalCount"] == 2


class TestToolGalleryProvidersRoute:
    def test_returns_both_providers_with_labels(self) -> None:
        # GIVEN the gallery routes
        mcp = _make_server_with_route()
        # WHEN the providers enum route is requested
        with TestClient(mcp.http_app()) as client:
            resp = client.get("/toolGallery/providers/")
        # THEN both providers are returned as {value, label}, count matching
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == len(body["providers"]) == 2
        by_value = {item["value"]: item["label"] for item in body["providers"]}
        assert by_value == {"datarobot": "DataRobot", "third_party": "Third party"}


class TestToolGalleryEnumRoutesAreGated:
    def test_denied_gate_returns_404_for_enum_routes(self) -> None:
        # GIVEN a gallery whose gate denies access
        async def deny(_request: Any) -> bool:
            return False

        mcp = _make_server_with_route_gated(deny)
        # WHEN the enum routes are requested
        with TestClient(mcp.http_app()) as client:
            categories = client.get("/toolGallery/categories/")
            providers = client.get("/toolGallery/providers/")
        # THEN both are hidden (404) just like /tools/
        assert categories.status_code == 404
        assert providers.status_code == 404

    def test_custom_base_path_is_honored_for_enum_routes(self) -> None:
        # GIVEN a gallery mounted under a custom prefix
        mcp = FastMCP("custom-path-enums")
        register_tool_gallery_routes(mcp, base_path="/prefixed/toolGallery")
        # WHEN the enum routes are requested at the prefixed and bare paths
        with TestClient(mcp.http_app()) as client:
            # THEN they answer only under the configured prefix
            assert client.get("/prefixed/toolGallery/categories/").status_code == 200
            assert client.get("/prefixed/toolGallery/providers/").status_code == 200
            assert client.get("/toolGallery/categories/").status_code == 404


class TestUiMetadataProvider:
    """The route re-attaches UI fields from the injected ``ui_metadata_provider``."""

    def _server(self, provider: Any) -> FastMCP:
        mcp = FastMCP("tool-gallery-ui")

        @mcp.tool
        def jira_search_issues(a: int) -> int:
            """Search."""
            return a

        register_tool_gallery_routes(mcp, ui_metadata_provider=provider)
        return mcp

    def test_provider_fields_are_surfaced(self) -> None:
        def provider() -> dict[str, dict[str, Any]]:
            return {
                "jira_search_issues": {
                    "display_name": "Jira — Search Issues",
                    "description_ui": "Find Jira issues matching a JQL query.",
                    "auth_provider": "jira",
                }
            }

        mcp = self._server(provider)
        with TestClient(mcp.http_app()) as client:
            body = client.get("/toolGallery/tools/").json()
        item = {t["name"]: t for t in body["tools"]}["jira_search_issues"]
        assert item["display_name"] == "Jira — Search Issues"
        assert item["description"] == "Find Jira issues matching a JQL query."
        assert item["provider"] == "third_party"
        assert item["oauth_provider_type"] == "jira"

    def test_missing_tool_in_provider_falls_back_to_defaults(self) -> None:
        # Provider returns nothing for this tool → UI fields default, provider=datarobot.
        # description has no curated description_ui, so it falls back to the MCP description.
        mcp = self._server(lambda: {})
        with TestClient(mcp.http_app()) as client:
            body = client.get("/toolGallery/tools/").json()
        item = {t["name"]: t for t in body["tools"]}["jira_search_issues"]
        assert item["display_name"] == "jira_search_issues"
        assert item["description"] == "Search."
        assert item["provider"] == "datarobot"
        assert item["oauth_provider_type"] is None


class TestToolGalleryGate:
    def test_gate_allows_serves_catalog(self) -> None:
        async def allow(_request: Any) -> bool:
            return True

        mcp = _make_server_with_route_gated(allow)
        with TestClient(mcp.http_app()) as client:
            resp = client.get("/toolGallery/tools/")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_gate_denies_returns_404(self) -> None:
        async def deny(_request: Any) -> bool:
            return False

        mcp = _make_server_with_route_gated(deny)
        with TestClient(mcp.http_app()) as client:
            resp = client.get("/toolGallery/tools/")
        assert resp.status_code == 404

    def test_gate_raising_fails_closed_to_404(self) -> None:
        async def boom(_request: Any) -> bool:
            raise RuntimeError("flag service down")

        mcp = _make_server_with_route_gated(boom)
        with TestClient(mcp.http_app()) as client:
            resp = client.get("/toolGallery/tools/")
        assert resp.status_code == 404

    def test_gate_receives_request_headers(self) -> None:
        seen: dict[str, str] = {}

        async def capture(request: Any) -> bool:
            seen["token"] = request.headers.get("x-datarobot-authorization", "")
            return True

        mcp = _make_server_with_route_gated(capture)
        with TestClient(mcp.http_app()) as client:
            client.get("/toolGallery/tools/", headers={"x-datarobot-authorization": "Bearer tok"})
        assert seen["token"] == "Bearer tok"
