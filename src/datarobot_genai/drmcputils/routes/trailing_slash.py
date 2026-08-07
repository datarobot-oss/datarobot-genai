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

This lives in ``drmcputils`` — beside the registrars whose paths it normalizes, and the
one package every server and sibling package may import. It was previously a
``drmcp``-private middleware covering only the MCP mount, which put it out of reach of
global-mcp (which imports nothing from ``drmcp``) and left every REST route redirecting on
both servers.

Rules are ``(prefix, canonical trailing slash)`` and are matched against the request path,
longest prefix first. A prefix form rather than exact paths is deliberate: it covers
parameterized routes such as ``/toolGallery/toolSets/{id}/``, which an exact-path table
would miss — and that is the route a UI hits most.

**Build the rules from the same function that built the routes.** ``shared_route_slash_rules``
takes the caller's path-prefixing callable so a mounted server (``URL_PREFIX`` set — the
deployment-backed shape) and an unmounted one (no prefix — the workload-backed shape, and
global-mcp) both produce rules that match their own registered paths. Hard-coding the
strings here would silently stop matching the moment a server is mounted.
"""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence

from starlette.types import ASGIApp
from starlette.types import Receive
from starlette.types import Scope
from starlette.types import Send

# (path prefix, whether the registered spelling carries a trailing slash).
SlashRule = tuple[str, bool]

# Applies a server's mount prefix to a bare route path. ``drmcp``'s ``prefix_mount_path``
# is one; the identity function is the unmounted case.
PathPrefixer = Callable[[str], str]

# FastMCP's streamable-http mount. Registered without a trailing slash on both servers.
DEFAULT_MCP_PATH = "/mcp"


def shared_route_slash_rules(prefix: PathPrefixer | None = None) -> tuple[SlashRule, ...]:
    """Rules for the REST route groups this package registers.

    ``/metadata`` is the odd one out — registered bare, to match the shape user-mcp's own
    inline route has always had. With normalization in place that asymmetry stops being a
    contract callers have to know: both spellings work either way.
    """
    # Imported here rather than at module scope: the registrars import this module for
    # nothing, but a future one might, and route modules importing each other at import
    # time is a cycle waiting to happen.
    from datarobot_genai.drmcputils.routes.metadata import METADATA_BASE_PATH
    from datarobot_genai.drmcputils.routes.tool_gallery import TOOL_GALLERY_BASE_PATH

    apply = prefix if prefix is not None else (lambda path: path)
    # One prefix covers every /toolGallery sub-route — tools, categories, providers and
    # global-mcp's toolSets/{id}. That is why the rules are prefixes and not exact paths:
    # a route added to the group inherits normalization by existing.
    return (
        (apply(TOOL_GALLERY_BASE_PATH), True),
        (apply(METADATA_BASE_PATH), False),
    )


def mcp_slash_rule(prefix: PathPrefixer | None = None) -> SlashRule:
    """Rule for the streamable-http MCP mount."""
    apply = prefix if prefix is not None else (lambda path: path)
    return (apply(DEFAULT_MCP_PATH), False)


def default_slash_rules(prefix: PathPrefixer | None = None) -> tuple[SlashRule, ...]:
    """Build rules for the MCP mount plus every shared REST route group."""
    return (mcp_slash_rule(prefix), *shared_route_slash_rules(prefix))


class TrailingSlashNormalizer:
    """Rewrite a request's trailing slash to the spelling its route group registered.

    Install it **outermost**. Middleware that keys off the request path — user-mcp's
    ``RequestHeadersMiddleware`` skips the MCP mount that way — has to see the normalized
    path, or it makes its decision on a spelling the router will never route.
    """

    def __init__(self, app: ASGIApp, rules: Iterable[SlashRule]) -> None:
        self._app = app
        # Longest prefix first, so a rule for a nested group beats its parent's.
        self._rules: Sequence[SlashRule] = sorted(
            ((prefix.rstrip("/") or "/", slashed) for prefix, slashed in rules),
            key=lambda rule: len(rule[0]),
            reverse=True,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path")
        if scope.get("type") == "http" and isinstance(path, str):
            canonical = self._canonical(path)
            if canonical is not None and canonical != path:
                scope = dict(scope, path=canonical, raw_path=self._rewrite_raw(scope, canonical))
        await self._app(scope, receive, send)

    def _canonical(self, path: str) -> str | None:
        """Return the spelling a route registered for *path*, or ``None`` if no rule fits."""
        # Collapse repeated slashes too: `/mcp///` and `/mcp/` are the same request.
        bare = path.rstrip("/") or "/"
        for prefix, slashed in self._rules:
            if bare != prefix and not bare.startswith(prefix.rstrip("/") + "/"):
                continue
            if bare == "/":
                return None
            return bare + "/" if slashed else bare
        return None

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
        head = raw[: len(raw) - len(scope.get("path", "").encode("utf-8"))]
        return head + canonical.encode("utf-8")
