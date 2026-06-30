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

"""Tests for the shared ``GET /tools-gallery`` route."""

import importlib
import importlib.util
import pkgutil
from typing import Any

from fastmcp import FastMCP
from starlette.testclient import TestClient

from datarobot_genai.drmcputils.routes.tool_gallery import _is_hosted
from datarobot_genai.drmcputils.routes.tool_gallery import _merge_tool_info
from datarobot_genai.drmcputils.routes.tool_gallery import register_tool_gallery_routes
from datarobot_genai.drmcputils.tool_gallery import DRTOOLS_PRIVATE_METADATA_KEYS
from datarobot_genai.drtools.core import get_registered_tools


def _all_live_drtools_metadata() -> list[tuple[str, dict]]:
    """Import every drtools tool module and return (tool_name, metadata) pairs."""

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
    return [((md.get("name") or fn.__name__), md) for fn, md in get_registered_tools()]


class _FakeTool:
    def __init__(self, name: str, tags: set[str] | None = None, meta: dict | None = None) -> None:
        self.name = name
        self.tags = tags or set()
        self.meta = meta


class TestIsHosted:
    def test_user_tool_deployment_is_hosted(self) -> None:
        assert _is_hosted(_FakeTool("t", meta={"tool_category": "USER_TOOL_DEPLOYMENT"}))

    def test_built_in_tool_is_not_hosted(self) -> None:
        assert not _is_hosted(_FakeTool("t", meta={"tool_category": "BUILT_IN_TOOL"}))

    def test_no_meta_is_not_hosted(self) -> None:
        assert not _is_hosted(_FakeTool("t", meta=None))

    def test_empty_meta_is_not_hosted(self) -> None:
        assert not _is_hosted(_FakeTool("t", meta={}))


class TestMergeToolInfo:
    def test_merges_ui_metadata_and_derived_categories(self) -> None:
        tool = _FakeTool("jira_search_issues", tags={"jira", "search"})
        ui = {
            "jira_search_issues": {
                "display_name": "Jira — Search Issues",
                "description_ui": "Find Jira issues matching a JQL query.",
                "auth_provider": "jira",
            }
        }
        merged = _merge_tool_info(tool, ui)
        assert merged["name"] == "jira_search_issues"
        assert merged["display_name"] == "Jira — Search Issues"
        assert merged["description_ui"] == "Find Jira issues matching a JQL query."
        assert merged["auth_provider"] == "jira"
        assert merged["tags"] == ["jira", "search"]
        assert merged["categories"] == ["dr_connector_jira", "dr_connectors"]
        assert merged["hosted"] is False

    def test_tool_without_ui_metadata_gets_none_fields(self) -> None:
        merged = _merge_tool_info(_FakeTool("jira_search_issues"), {})
        assert merged["display_name"] is None
        assert merged["description_ui"] is None
        assert merged["auth_provider"] is None
        # Categories still derived from the static taxonomy.
        assert merged["categories"] == ["dr_connector_jira", "dr_connectors"]

    def test_hosted_tool_has_no_categories(self) -> None:
        tool = _FakeTool("user_xyz", meta={"tool_category": "USER_TOOL_DEPLOYMENT"})
        merged = _merge_tool_info(tool, {})
        assert merged["hosted"] is True
        assert merged["categories"] == []


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


def _make_server_with_route_gated(gate: Any) -> FastMCP:
    mcp = FastMCP("tool-gallery-gated")

    @mcp.tool
    def jira_search_issues(a: int) -> int:
        """Search."""
        return a

    register_tool_gallery_routes(mcp, gate=gate)
    return mcp


class TestDrtoolsUiMetadataCompleteness:
    """Every live drtools tool must carry the UI metadata the gallery surfaces."""

    def test_every_tool_has_display_name_and_description_ui(self) -> None:
        missing_display = []
        missing_desc = []
        for name, md in _all_live_drtools_metadata():
            if not md.get("display_name"):
                missing_display.append(name)
            if not md.get("description_ui"):
                missing_desc.append(name)
        assert not missing_display, f"tools missing display_name: {sorted(missing_display)}"
        assert not missing_desc, f"tools missing description_ui: {sorted(missing_desc)}"

    def test_connector_and_web_tools_have_auth_provider(self) -> None:
        needs_auth = {"confluence", "jira", "gdrive", "microsoft_graph", "perplexity", "tavily"}

        # Ensure modules are imported.
        _all_live_drtools_metadata()
        missing = []
        for fn, md in get_registered_tools():
            mod = fn.__module__
            pkg = mod.split(".")[3] if mod.startswith("datarobot_genai.drtools.") else ""
            if pkg in needs_auth and not md.get("auth_provider"):
                missing.append(md.get("name") or fn.__name__)
        assert not missing, f"connector/web tools missing auth_provider: {sorted(missing)}"


class TestUiMetadataNeverLeaksToMcpClient:
    """The UI fields must be in the strip set so agents/LLMs never see them."""

    def test_ui_keys_are_private(self) -> None:
        for key in ("display_name", "description_ui", "auth_provider"):
            assert key in DRTOOLS_PRIVATE_METADATA_KEYS, (
                f"{key!r} must be stripped before FastMCP registration"
            )
