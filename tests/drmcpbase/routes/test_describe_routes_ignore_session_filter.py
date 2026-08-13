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

"""The describe-the-server REST routes must not be reshaped by session headers.

``/toolGallery/tools/``, ``/toolGallery/categories/`` and ``/metadata`` answer "what can this
server do", and UIs build their pickers from them — a taxonomy already narrowed by the
caller's own filter cannot offer the categories that filter excluded.

``list_tools(run_middleware=False)`` does not deliver that on its own: in FastMCP the flag
suppresses middleware but registered catalog transforms still run, and DataRobot's
transform is exactly where mode/allowlist filtering lives. These tests pin the difference
between a server that passes ``unfiltered_catalog_provider`` and one that does not.
"""

from typing import Any

import pytest
from fastmcp import FastMCP
from starlette.testclient import TestClient

from datarobot_genai.drmcpbase.fastmcp_transforms import register_mcp_catalog_transform
from datarobot_genai.drmcpbase.fastmcp_transforms import unfiltered_catalog_provider
from datarobot_genai.drmcputils.routes import register_metadata_routes
from datarobot_genai.drmcputils.routes import register_tool_gallery_routes

_ONLY_ONE_TOOL = {"x-datarobot-mcp-tools": "jira_search_issues"}


def _server(*, unfiltered: bool) -> FastMCP:
    mcp = FastMCP("describe-routes-test")

    @mcp.tool
    def jira_search_issues(a: int) -> int:
        """Search issues."""
        return a

    @mcp.tool
    def perplexity_search(q: str) -> str:
        """Search the web."""
        return q

    register_mcp_catalog_transform(mcp)
    provider = unfiltered_catalog_provider(mcp) if unfiltered else None
    register_tool_gallery_routes(mcp, catalog_provider=provider)
    register_metadata_routes(mcp, catalog_provider=provider)
    return mcp


def _tool_names(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    return {
        "gallery": sorted(
            item["name"]
            for item in client.get("/toolGallery/tools/", headers=headers).json()["tools"]
        ),
        "metadata": sorted(
            item["name"]
            for item in client.get("/metadata", headers=headers).json()["tools"]["items"]
        ),
        "categories": client.get("/toolGallery/categories/", headers=headers).json()["totalCount"],
    }


class TestDescribeRoutesIgnoreTheSessionFilter:
    def test_an_allowlist_header_does_not_narrow_the_catalog(self) -> None:
        """GIVEN a caller allowlisted to one tool, THEN all three routes still report both."""
        with TestClient(_server(unfiltered=True).http_app()) as client:
            unfiltered = _tool_names(client, {})
            filtered = _tool_names(client, _ONLY_ONE_TOOL)

        assert unfiltered["gallery"] == ["jira_search_issues", "perplexity_search"]
        assert filtered == unfiltered

    @pytest.mark.parametrize("mode", ["search", "code"])
    def test_an_unsupported_or_collapsing_mode_header_does_not_change_the_answer(
        self, mode: str
    ) -> None:
        """``mode=search`` used to zero the counts; ``mode=code`` used to 500 all three."""
        with TestClient(_server(unfiltered=True).http_app()) as client:
            resp = client.get("/toolGallery/categories/", headers={"x-datarobot-mcp-mode": mode})
            gallery = client.get("/toolGallery/tools/", headers={"x-datarobot-mcp-mode": mode})

        assert resp.status_code == 200
        assert resp.json()["totalCount"] == 2
        assert gallery.status_code == 200
        assert len(gallery.json()["tools"]) == 2

    def test_without_a_provider_the_transform_still_narrows_them(self) -> None:
        """The regression this guards: the plain call is not transform-free.

        Kept as an explicit assertion rather than a comment — if a FastMCP upgrade ever
        makes ``run_middleware=False`` bypass transforms too, this test fails and the
        provider plumbing can be deleted.
        """
        with TestClient(_server(unfiltered=False).http_app()) as client:
            filtered = _tool_names(client, _ONLY_ONE_TOOL)

        assert filtered["gallery"] == ["jira_search_issues"]
        assert filtered["metadata"] == ["jira_search_issues"]
        assert filtered["categories"] == 1

    def test_the_gallery_answers_json_when_the_catalog_cannot_be_built(self) -> None:
        """A transform that rejects the request must not escape as a bare 500."""
        with TestClient(_server(unfiltered=False).http_app()) as client:
            resp = client.get("/toolGallery/tools/", headers={"x-datarobot-mcp-mode": "code"})

        assert resp.status_code == 500
        assert "error" in resp.json()
