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

"""Serve routes with or without their trailing slash — by rewriting, not redirecting.

Starlette answers a near-miss on the slash with a ``307`` whose ``Location`` is built from
the request as the *container* saw it. Behind DataRobot's deployment ``directAccess``
gateway that URL does not map back to anything the client can reach, so the redirect is a
dead end rather than a correction. Rewriting the path inside the ASGI scope serves both
spellings from the one registered route, and the client's single request is answered.

**The router is the source of truth for which spellings exist.** There is no table of
paths or prefixes to keep in step with the routes: the middleware asks the app's own
router whether the other spelling matches a registered route, and rewrites only if it
does. That is the same question ``redirect_slashes`` asks — this answers it with a
rewrite instead of a redirect.

A rules table was the previous design, and prefixes were what made it dangerous.
``("/toolGallery", True)`` asserted a fact about every route under that prefix,
including ones not yet written: a sibling registered *without* a trailing slash would
be rewritten to a spelling the router does not serve, Starlette would ``307`` back to
the spelling the middleware rejects, and the two would bounce forever — strictly worse
than the ``404`` you would get with no middleware at all. Asking the router cannot
produce that, because it never rewrites to a path that does not exist. It also needs no
maintenance when a route is added, and covers routes the table never listed (FastMCP's
own mount, ``/.well-known/*``, health).
"""

import logging

from starlette.routing import Match
from starlette.types import ASGIApp
from starlette.types import Receive
from starlette.types import Scope
from starlette.types import Send

logger = logging.getLogger(__name__)


class TrailingSlashNormalizer:
    """Rewrite a request's trailing slash to the spelling its route actually registered.

    Install it **outermost**. Middleware that keys off the request path — user-mcp's
    ``RequestHeadersMiddleware`` skips the MCP mount that way — has to see the normalized
    path, or it makes its decision on a spelling the router will never route.

    Outermost is also where ``scope["app"]`` is already set: Starlette assigns it before
    entering the middleware stack, so the router is reachable from the very first layer.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self._warned_no_router = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http" and isinstance(scope.get("path"), str):
            canonical = self._canonical(scope)
            if canonical is not None:
                scope = dict(scope, path=canonical, raw_path=self._rewrite_raw(scope, canonical))
        await self._app(scope, receive, send)

    def _canonical(self, scope: Scope) -> str | None:
        """Return the registered spelling for this path, or ``None`` to leave it alone.

        ``None`` covers every case where rewriting would be wrong or pointless: the path
        already routes, no other spelling routes either (let the router 404 on what the
        client actually sent), or the router is unreachable.
        """
        path: str = scope["path"]
        routes = self._routes(scope)
        if routes is None:
            return None
        if self._matches(routes, scope, path):
            return None
        # Collapse repeated trailing slashes too: `/mcp///`, `/mcp/` and `/mcp` are one
        # request as far as any route table is concerned.
        bare = path.rstrip("/") or "/"
        for candidate in (bare, bare + "/"):
            if candidate != path and self._matches(routes, scope, candidate):
                return candidate
        return None

    def _routes(self, scope: Scope) -> list | None:
        """Return the app's route table, or ``None`` (warned once) if unreachable."""
        routes = getattr(scope.get("app"), "routes", None)
        if routes is None and not self._warned_no_router:
            self._warned_no_router = True
            logger.warning(
                "TrailingSlashNormalizer cannot reach the app router (no scope['app']); "
                "trailing-slash near-misses will fall through to Starlette's redirect. "
                "Install it as Starlette middleware so the router is in scope."
            )
        return routes

    @staticmethod
    def _matches(routes: list, scope: Scope, path: str) -> bool:
        """Whether *path* would route, ignoring the method.

        ``Match.PARTIAL`` — right path, wrong method — counts: a ``POST`` to the
        slash-less spelling of a ``GET`` route should be normalized and then answered
        ``405`` by the router, not left to miss the route entirely.
        """
        probe = dict(scope, path=path)
        return any(route.matches(probe)[0] is not Match.NONE for route in routes)

    @staticmethod
    def _rewrite_raw(scope: Scope, canonical: str) -> bytes:
        """Rebuild ``raw_path`` around the new path, preserving any ``root_path`` prefix.

        Assigning the bare path here would drop the prefix uvicorn includes in
        ``raw_path``, which is a quiet ASGI-convention violation even where — as with
        Starlette, which routes on ``path`` — nothing downstream currently reads it.
        """
        raw = scope.get("raw_path")
        root = scope.get("root_path") or ""
        if not isinstance(raw, bytes):
            return (root + canonical).encode("utf-8")
        # raw_path is the undecoded target; keep whatever precedes the routed path.
        # A percent-encoded target makes `path` shorter than its raw form, so only trust
        # the arithmetic when the decoded path really is a suffix of raw_path.
        tail = scope.get("path", "").encode("utf-8")
        if not raw.endswith(tail):
            return raw
        return raw[: len(raw) - len(tail)] + canonical.encode("utf-8")
