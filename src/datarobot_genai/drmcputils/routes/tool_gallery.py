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

"""``/toolGallery/*`` — rich tool metadata routes for UIs, shared by both servers.

The first (and currently only) route is ``GET /toolGallery/tools/``. The group is
designed to grow (e.g. ``/toolGallery/categories/``), and **every** route under
``/toolGallery`` is gated by the same predicate — see ``register_tool_gallery_routes``.

The MCP ``tools/list`` response is intentionally lean: agents/LLMs never see the
UI-oriented fields (``display_name``, ``description_ui``, ``auth_provider``) — they
are stripped before FastMCP registration (see ``DRTOOLS_PRIVATE_METADATA_KEYS``).
The ``tools`` route re-attaches them via an injected ``ui_metadata_provider`` (drtools'
``get_tool_ui_metadata``) — injected, not imported, because ``drmcputils`` may not import
``drtools`` — and derives each tool's categories from the single-source-of-truth taxonomy.
It returns the full catalog (not the per-request filtered/CodeMode view), paginated via
``limit``/``offset``.
"""

import logging
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from datarobot_genai.drmcputils.tool_gallery import build_tool_gallery_items
from datarobot_genai.drmcputils.tool_gallery import merge_tool_info

logger = logging.getLogger(__name__)

# Base path for the route group. Singular "toolGallery"; sub-routes hang off it
# (e.g. /toolGallery/tools/). All sub-routes are gated together.
_DEFAULT_BASE_PATH = "/toolGallery"
# Default page size when the request omits ``limit``.
_DEFAULT_LIMIT = 100

# An optional per-request access gate. Returns True to serve the gallery, False to
# hide it (404). global-mcp injects a feature-flag check; user-mcp leaves it unset.
ToolGalleryGate = Callable[[Request], Awaitable[bool]]

# Supplies ``tool_name -> {display_name, description_ui, auth_provider}``. Injected by the
# caller (drtools' ``get_tool_ui_metadata``) so this module need not import drtools. When
# unset, the gallery serves names/tags/categories only (UI fields fall back to defaults).
UiMetadataProvider = Callable[[], dict[str, dict[str, Any]]]


def _parse_pagination(request: Request) -> tuple[int, int]:
    """Read ``limit`` (default 100) and ``offset`` (default 0) from the query string.

    Non-integer or negative values fall back to the defaults, so a malformed query never
    500s the gallery.
    """

    def _non_negative_int(name: str, default: int) -> int:
        raw = request.query_params.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return value if value >= 0 else default

    return _non_negative_int("limit", _DEFAULT_LIMIT), _non_negative_int("offset", 0)


def register_tool_gallery_routes(
    mcp: Any,
    base_path: str = _DEFAULT_BASE_PATH,
    gate: ToolGalleryGate | None = None,
    ui_metadata_provider: UiMetadataProvider | None = None,
) -> None:
    """Register every ``/toolGallery/*`` route on the FastMCP server, all sharing *gate*.

    ``tools`` is the first route (``GET <base_path>/tools/``). New gallery routes are
    added in one place here and inherit the same gating automatically — the gate is
    applied uniformly to the whole group rather than wired per route at the call site.

    Args:
        mcp: FastMCP server instance.
        base_path: Group prefix. Configurable so a mounted server (user-mcp) can prefix
            it via ``prefix_mount_path`` while global-mcp uses the bare ``/toolGallery``.
        gate: Optional async predicate ``(Request) -> bool`` applied to every route in
            the group. When it returns ``False`` (or raises), the route responds ``404``
            so the feature stays hidden. Both servers pass a feature-flag gate.
        ui_metadata_provider: Optional ``() -> {tool_name: {display_name, description_ui,
            auth_provider}}`` callable supplying the UI-only fields. Injected (rather than
            imported) so this module avoids a forbidden ``drmcputils -> drtools`` import;
            both servers pass drtools' ``get_tool_ui_metadata``. When unset, those fields
            fall back to defaults (provider classified as ``datarobot``).
    """
    prefix = base_path.rstrip("/")

    # (sub-path, handler) for each route in the group. Add new gallery routes here;
    # they are gated identically. ``tools`` is the first.
    routes: list[tuple[str, Callable[[Request], Awaitable[JSONResponse]]]] = [
        ("/tools/", _make_tools_handler(mcp, ui_metadata_provider)),
    ]
    for sub_path, handler in routes:
        _register_gated_route(mcp, f"{prefix}{sub_path}", gate, handler)

    logger.info(
        "toolGallery routes registered under %s (%d route(s), gated=%s)",
        prefix,
        len(routes),
        gate is not None,
    )


def _register_gated_route(
    mcp: Any,
    path: str,
    gate: ToolGalleryGate | None,
    handler: Callable[[Request], Awaitable[JSONResponse]],
) -> None:
    """Register a single ``GET`` route that runs *gate* (fail-closed) before *handler*."""

    @mcp.custom_route(path, methods=["GET"])
    async def route(request: Request) -> JSONResponse:
        if gate is not None and not await _gate_allows(gate, request):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return await handler(request)


def _make_tools_handler(
    mcp: Any,
    ui_metadata_provider: UiMetadataProvider | None,
) -> Callable[[Request], Awaitable[JSONResponse]]:
    """Build the ``/tools/`` handler bound to *mcp* (full paginated tool catalog)."""

    async def tools_handler(request: Request) -> JSONResponse:
        # run_middleware=False → the full catalog, not the per-request
        # allowlist-filtered / CodeMode-collapsed view. The gallery shows everything.
        tools = await mcp.list_tools(run_middleware=False)
        ui_metadata = ui_metadata_provider() if ui_metadata_provider is not None else {}
        merged = [merge_tool_info(tool, ui_metadata) for tool in tools]
        items = build_tool_gallery_items(merged)

        total_count = len(items)
        limit, offset = _parse_pagination(request)
        page = items[offset : offset + limit]
        return JSONResponse(
            {
                "tools": page,
                "count": len(page),
                "totalCount": total_count,
                "limit": limit,
                "offset": offset,
                "hasMore": offset + len(page) < total_count,
            }
        )

    return tools_handler


async def _gate_allows(gate: ToolGalleryGate, request: Request) -> bool:
    """Evaluate *gate*, treating any failure as "deny" (fail closed)."""
    try:
        return await gate(request)
    except Exception:
        logger.warning("toolGallery gate raised; denying access", exc_info=True)
        return False
