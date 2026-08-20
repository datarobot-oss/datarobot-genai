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

The normalizer reads the registered spellings off the app's own router, so these tests
drive it through real Starlette apps rather than a table of rules.
"""

from typing import Any

import pytest
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.routing import Mount
from starlette.routing import Route
from starlette.testclient import TestClient

from datarobot_genai.drmcputils.routes import TrailingSlashNormalizer
from datarobot_genai.drmcputils.routes import register_metadata_routes
from datarobot_genai.drmcputils.routes import register_tool_gallery_routes


def _ok(_request: Any) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _app(
    paths: list[str], methods: list[str] | None = None, seen: list[str] | None = None
) -> Starlette:
    """Build a Starlette app serving exactly *paths*, with the normalizer outermost.

    When *seen* is given, each endpoint records the path it was reached at — which is
    the post-rewrite spelling, i.e. what the route actually matched.
    """

    async def endpoint(request: Any) -> PlainTextResponse:
        if seen is not None:
            seen.append(request.scope["path"])
        return PlainTextResponse("ok")

    return Starlette(
        routes=[Route(p, endpoint, methods=methods or ["GET"]) for p in paths],
        middleware=[Middleware(TrailingSlashNormalizer)],
    )


def _reaches(paths: list[str], requested: str) -> tuple[int, str | None]:
    """Request *requested* against an app serving *paths*; report (status, matched path)."""
    seen: list[str] = []
    with TestClient(_app(paths, seen=seen)) as client:
        resp = client.get(requested, follow_redirects=False)
    return resp.status_code, (seen[0] if seen else None)


class TestBothSpellingsAreServed:
    """Whichever spelling is registered, the other one reaches it — as a rewrite."""

    @pytest.mark.parametrize(
        "registered, requested",
        [
            # Registered bare (FastMCP's /mcp mount is an exact Route like this).
            ("/mcp", "/mcp"),
            ("/mcp", "/mcp/"),
            ("/mcp", "/mcp///"),
            ("/metadata", "/metadata"),
            ("/metadata", "/metadata/"),
            # Registered slashed (every /toolGallery REST route).
            ("/toolGallery/tools/", "/toolGallery/tools"),
            ("/toolGallery/tools/", "/toolGallery/tools/"),
            ("/toolGallery/categories/", "/toolGallery/categories"),
        ],
    )
    def test_request_reaches_the_registered_route(self, registered: str, requested: str) -> None:
        status, seen = _reaches([registered], requested)
        assert status == 200, f"{requested} → {status}"
        assert seen == registered

    def test_parameterized_routes_normalize_too(self) -> None:
        # The route a UI hits most, and the one an exact-path table would have missed.
        status, seen = _reaches(["/toolGallery/toolSets/{sid}/"], "/toolGallery/toolSets/665f")
        assert status == 200
        assert seen == "/toolGallery/toolSets/665f/"

    def test_a_wrong_method_still_normalizes_then_405s(self) -> None:
        # Match.PARTIAL — right path, wrong method — must still be rewritten, so the
        # router answers 405 for the route rather than missing it entirely.
        app = _app(["/toolGallery/tools/"], methods=["GET"])
        with TestClient(app) as client:
            resp = client.post("/toolGallery/tools", follow_redirects=False)
        assert resp.status_code == 405


class TestItCannotCreateARedirectLoop:
    """The failure the rules table made possible, and this design cannot.

    ``("/toolGallery", True)`` forced a trailing slash onto every path under the prefix,
    including routes registered without one. Starlette's ``redirect_slashes`` stripped it
    straight back and the two bounced forever — worse than the 404 you get with no
    middleware. Rewriting only to a spelling the router actually serves rules that out.
    """

    def test_a_slashless_route_beside_slashed_siblings_is_left_alone(self) -> None:
        paths = ["/toolGallery/tools/", "/toolGallery/toolSets/{sid}"]
        # The slash-less sibling is served as registered...
        status, seen = _reaches(paths, "/toolGallery/toolSets/abc")
        assert status == 200
        assert seen == "/toolGallery/toolSets/abc"
        # ...and its slashed spelling resolves onto it rather than looping.
        status, seen = _reaches(paths, "/toolGallery/toolSets/abc/")
        assert status == 200
        assert seen == "/toolGallery/toolSets/abc"

    def test_an_unknown_path_is_not_rewritten(self) -> None:
        # Neither spelling routes, so the client's own path reaches the 404 — no
        # invented rewrite, and nothing to bounce against.
        with TestClient(_app(["/metadata"])) as client:
            resp = client.get("/nope/", follow_redirects=False)
        assert resp.status_code == 404


class TestMountedUnderAPrefix:
    """The deployment-backed user MCP: ``URL_PREFIX`` set, every path prefixed.

    Nothing to configure — the router holds the prefixed paths, so they are what the
    normalizer reads.
    """

    @pytest.mark.parametrize(
        "requested, expected",
        [
            ("/api/mcp/", "/api/mcp"),
            ("/api/toolGallery/tools", "/api/toolGallery/tools/"),
        ],
    )
    def test_prefixed_paths_normalize(self, requested: str, expected: str) -> None:
        status, seen = _reaches(["/api/mcp", "/api/toolGallery/tools/"], requested)
        assert status == 200
        assert seen == expected

    def test_the_unprefixed_spelling_is_somebody_elses_route(self) -> None:
        with TestClient(_app(["/api/mcp"])) as client:
            assert client.get("/mcp/", follow_redirects=False).status_code == 404


class TestScopeHandling:
    async def test_non_http_scopes_pass_through_by_identity(self) -> None:
        """A lifespan scope must reach the app untouched, not merely equal."""
        seen: list[Any] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            seen.append(scope)

        lifespan = {"type": "lifespan"}
        await TrailingSlashNormalizer(app)(lifespan, None, None)
        assert seen[0] is lifespan

    async def test_no_router_in_scope_leaves_the_path_alone(self) -> None:
        # Installed outside Starlette there is no scope["app"]; degrade to leaving the
        # path untouched (and warn) rather than guessing at spellings.
        seen: list[Any] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            seen.append(scope)

        scope = {"type": "http", "path": "/toolGallery/tools"}
        await TrailingSlashNormalizer(app)(scope, None, None)
        assert seen[0] is scope

    def test_raw_path_keeps_its_root_path_prefix(self) -> None:
        """Assigning the bare path would drop the prefix uvicorn puts in ``raw_path``."""
        seen: list[bytes] = []

        async def record(request: Any) -> PlainTextResponse:
            seen.append(request.scope["raw_path"])
            return PlainTextResponse("ok")

        app = Starlette(
            routes=[Route("/mcp", record)], middleware=[Middleware(TrailingSlashNormalizer)]
        )
        with TestClient(app, root_path="/root") as client:
            client.get("/mcp/", follow_redirects=False)
        assert seen and seen[0].endswith(b"/mcp")
        assert not seen[0].endswith(b"/mcp/")


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
        return TestClient(mcp.http_app(middleware=[Middleware(TrailingSlashNormalizer)]))

    @pytest.mark.parametrize(
        "path",
        [
            "/toolGallery/tools",
            "/toolGallery/tools/",
            "/toolGallery/categories",
            "/toolGallery/categories/",
            "/toolGallery/providers",
            "/toolGallery/providers/",
            "/metadata",
            "/metadata/",
        ],
    )
    def test_both_spellings_answer_without_a_redirect(self, path: str) -> None:
        with self._client() as client:
            resp = client.get(path, follow_redirects=False)
        assert resp.status_code == 200, f"{path} → {resp.status_code}"


class TestMountedSubApps:
    """A ``Mount`` is a route too — the normalizer sees it without being told."""

    def test_paths_under_a_mount_are_untouched(self) -> None:
        inner = Starlette(routes=[Route("/thing/", _ok)])
        app = Starlette(
            routes=[Mount("/sub", inner)], middleware=[Middleware(TrailingSlashNormalizer)]
        )
        with TestClient(app) as client:
            # The mount matches both spellings itself, so the outer layer leaves the
            # path alone and the inner app routes it.
            assert client.get("/sub/thing/", follow_redirects=False).status_code == 200
