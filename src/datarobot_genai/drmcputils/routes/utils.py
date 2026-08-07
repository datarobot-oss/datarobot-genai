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

"""Shared gating helper for ``drmcputils`` HTTP route groups.

A route *gate* is an async predicate ``(Request) -> bool``. When it returns
``False`` — or raises — the route responds ``404`` so the feature stays hidden
(fail closed). global-mcp injects a per-user feature-flag check; user-mcp injects
the static-account check; either may pass ``None`` to leave a route open.

The tool-gallery route group predates this helper and keeps its own copy; new
route groups (``tool_gallery``, ``metadata``) share this one.
"""

import logging
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Async predicate deciding whether a request may see a route (True) or gets 404.
RouteGate = Callable[[Request], Awaitable[bool]]
# A route handler: request in, JSON response out.
RouteHandler = Callable[[Request], Awaitable[JSONResponse]]
# Supplies the tool catalog these routes describe. See `resolve_catalog`.
CatalogProvider = Callable[[], Awaitable[Sequence[Any]]]


async def resolve_catalog(mcp: Any, provider: CatalogProvider | None) -> Sequence[Any]:
    """Return the tool catalog a describe-the-server route should report on.

    These routes answer "what can this server do", not "what may this request call", so
    they must not be reshaped by the caller's ``x-datarobot-mcp-*`` session headers — a
    picker built from a filtered taxonomy cannot offer the categories you filtered out.

    ``run_middleware=False`` does *not* achieve that on its own: in FastMCP it suppresses
    middleware but still runs registered catalog transforms, and DataRobot's transform is
    where the mode/allowlist/gate filtering lives. So a caller that reuses one HTTP client
    for its MCP session and these routes gets a taxonomy narrowed to its own allowlist,
    all-zero counts under ``mode=search``, and a 500 under ``mode=code``.

    Servers that install the transform therefore pass a *provider* that resolves the
    catalog with the session filter neutralized (``drmcpbase.fastmcp_transforms``
    supplies one). Without a provider this falls back to the plain call — correct for any
    server with no transform installed, and unchanged from the previous behaviour.
    """
    if provider is not None:
        return await provider()
    return await mcp.list_tools(run_middleware=False)


def register_gated_get(
    mcp: Any,
    path: str,
    handler: RouteHandler,
    gate: RouteGate | None = None,
) -> None:
    """Register a single ``GET`` *path* that runs *gate* (fail closed) before *handler*."""

    @mcp.custom_route(path, methods=["GET"])
    async def route(request: Request) -> JSONResponse:
        if gate is not None and not await _gate_allows(gate, request):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return await handler(request)


async def _gate_allows(gate: RouteGate, request: Request) -> bool:
    """Evaluate *gate*, treating any failure as "deny" (fail closed)."""
    try:
        return await gate(request)
    except Exception:
        logger.warning("route gate raised; denying access", exc_info=True)
        return False
