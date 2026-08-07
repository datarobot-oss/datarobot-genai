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

"""Both spellings of every registered route must be served, without a redirect.

A ``307`` here carries a ``Location`` built from the path as the container saw it, which
does not resolve back through the deployment directAccess gateway — so the client
dead-ends rather than being corrected.
"""

from typing import Any

import pytest
from fastmcp import FastMCP
from starlette.testclient import TestClient

from datarobot_genai.drmcputils.routes import TrailingSlashNormalizer
from datarobot_genai.drmcputils.routes import default_slash_rules
from datarobot_genai.drmcputils.routes import register_metadata_routes
from datarobot_genai.drmcputils.routes import register_tool_gallery_routes


def _recording_app(seen: list[Any]):
    async def app(scope, receive, send):  # type: ignore[no-untyped-def]
        seen.append(scope)

    return app


async def _run(rules, scope):  # type: ignore[no-untyped-def]
    seen: list[Any] = []
    await TrailingSlashNormalizer(_recording_app(seen), rules)(scope, None, None)
    return seen[0]


class TestUnmountedServer:
    """global-mcp, and the workload-backed user MCP: no ``URL_PREFIX``, bare paths."""

    @pytest.mark.parametrize(
        "requested, expected",
        [
            # The MCP mount is registered bare; every spelling collapses onto it.
            ("/mcp", "/mcp"),
            ("/mcp/", "/mcp"),
            ("/mcp///", "/mcp"),
            # REST groups are registered slashed; the bare spelling gains one.
            ("/toolGallery/tools", "/toolGallery/tools/"),
            ("/toolGallery/tools/", "/toolGallery/tools/"),
            ("/toolGallery/categories", "/toolGallery/categories/"),
            ("/toolGallery/categories/", "/toolGallery/categories/"),
            # Parameterized routes are why the rules are prefixes, not exact paths.
            ("/toolGallery/toolSets/665f", "/toolGallery/toolSets/665f/"),
            ("/toolGallery/toolSets/665f/", "/toolGallery/toolSets/665f/"),
            # /metadata is registered bare, matching user-mcp's own inline route.
            ("/metadata", "/metadata"),
            ("/metadata/", "/metadata"),
            # Nothing outside a known group is touched, even under the MCP mount.
            ("/mcp/metadata", "/mcp/metadata"),
            ("/health/", "/health/"),
            ("/", "/"),
        ],
    )
    async def test_paths_resolve_to_the_registered_spelling(self, requested, expected) -> None:
        scope = await _run(default_slash_rules(), {"type": "http", "path": requested})
        assert scope["path"] == expected


class TestMountedServer:
    """The deployment-backed user MCP: ``URL_PREFIX`` set, every path prefixed.

    Rules come from the same prefixing callable the routes were registered with, so a
    mounted server matches its own paths and not the bare ones.
    """

    @staticmethod
    def _prefix(path: str) -> str:
        return "/api" + path

    @pytest.mark.parametrize(
        "requested, expected",
        [
            ("/api/mcp/", "/api/mcp"),
            ("/api/toolGallery/tools", "/api/toolGallery/tools/"),
            ("/api/metadata/", "/api/metadata"),
            # The unprefixed spelling is somebody else's route, not ours.
            ("/mcp/", "/mcp/"),
            ("/toolGallery/tools", "/toolGallery/tools"),
        ],
    )
    async def test_rules_follow_the_mount_prefix(self, requested, expected) -> None:
        scope = await _run(default_slash_rules(self._prefix), {"type": "http", "path": requested})
        assert scope["path"] == expected


class TestScopeHandling:
    async def test_raw_path_keeps_its_root_path_prefix(self) -> None:
        """Assigning the bare path would drop the prefix uvicorn puts in ``raw_path``."""
        scope = await _run(
            default_slash_rules(),
            {"type": "http", "path": "/mcp/", "raw_path": b"/root/mcp/", "root_path": "/root"},
        )
        assert scope["raw_path"] == b"/root/mcp"

    async def test_raw_path_is_synthesized_when_absent(self) -> None:
        scope = await _run(default_slash_rules(), {"type": "http", "path": "/mcp/"})
        assert scope["raw_path"] == b"/mcp"

    async def test_non_http_scopes_pass_through_by_identity(self) -> None:
        """A lifespan scope must reach the app untouched, not merely equal."""
        seen: list[Any] = []
        lifespan = {"type": "lifespan"}
        await TrailingSlashNormalizer(_recording_app(seen), default_slash_rules())(
            lifespan, None, None
        )
        assert seen[0] is lifespan

    async def test_a_matching_path_is_not_copied_needlessly(self) -> None:
        seen: list[Any] = []
        already_canonical = {"type": "http", "path": "/mcp"}
        await TrailingSlashNormalizer(_recording_app(seen), default_slash_rules())(
            already_canonical, None, None
        )
        assert seen[0] is already_canonical


class TestEndToEnd:
    """The point of the exercise: both spellings answer 200, and neither redirects."""

    @staticmethod
    def _client() -> TestClient:
        mcp = FastMCP("slash-e2e")

        @mcp.tool
        def vdb_list() -> int:
            """List."""
            return 1

        register_tool_gallery_routes(mcp)
        register_metadata_routes(mcp)
        app = mcp.http_app()
        return TestClient(TrailingSlashNormalizer(app, default_slash_rules()))

    @pytest.mark.parametrize(
        "path",
        [
            "/toolGallery/tools",
            "/toolGallery/tools/",
            "/toolGallery/categories",
            "/toolGallery/categories/",
            "/metadata",
            "/metadata/",
        ],
    )
    def test_both_spellings_answer_without_a_redirect(self, path: str) -> None:
        with self._client() as client:
            resp = client.get(path, follow_redirects=False)
        assert resp.status_code == 200, f"{path} → {resp.status_code}"
