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

Routes in the group:
  - ``GET /toolGallery/tools/`` — the full paginated tool catalog.
  - ``GET /toolGallery/categories/`` — the tool-category filter enum (``value`` + ``label``).
  - ``GET /toolGallery/providers/`` — the tool-provider filter enum (``value`` + ``label``).

The two enum routes back the UI filter panel: they return the filterable values (the same
``dr_*`` categories / ``datarobot``|``third_party`` providers a tool item carries) paired
with display labels, so the FE renders filters from the backend instead of hardcoding them.
The group is designed to grow, and **every** route under ``/toolGallery`` is gated by the
same predicate — see ``register_tool_gallery_routes``.

The MCP ``tools/list`` response is intentionally lean: agents/LLMs never see the
UI-oriented fields (``display_name``, ``description_ui``, ``auth_provider``) — they
are stripped before FastMCP registration (see ``DRTOOLS_PRIVATE_METADATA_KEYS``).
The ``tools`` route re-attaches them via an injected ``ui_metadata_provider`` (drtools'
``get_tool_ui_metadata``) — injected, not imported, because ``drmcputils`` may not import
``drtools`` — and derives each tool's categories from the single-source-of-truth taxonomy.
It returns the full catalog (not the per-request filtered/CodeMode view), optionally
filtered by ``name`` (exact), ``provider`` (``datarobot``/``third_party``) and ``category``
(a ``dr_*`` gallery category) — ``provider`` and ``category`` are multi-valued (comma-
separated and/or repeated params) and match-any within a dimension — and paginated via
``limit``/``offset``.
"""

import logging
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from datarobot_genai.drmcputils.categories import TOOL_CATEGORY_LABELS
from datarobot_genai.drmcputils.tool_gallery import TOOL_PROVIDER_LABELS
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


def _parse_filters(
    request: Request,
) -> tuple[str | None, list[str] | None, list[str] | None]:
    """Read the optional ``name``, ``provider`` and ``category`` filters from the query string.

    ``name`` is a single exact match. ``provider`` and ``category`` are **multi-valued**,
    mirroring the multi-select checkboxes in the gallery filter panel. Both forms are
    accepted and combined, so the value survives however a client serialises a list:
      - comma-separated: ``?category=dr_connectors,dr_web_search`` (recommended — the FE
        passes a plain ``values.join(",")`` string, avoiding array-serialisation quirks such
        as axios' default ``category[]=`` bracket keys); matches the repo's existing
        ``x-datarobot-mcp-tools`` header convention, and
      - repeated params: ``?category=dr_connectors&category=dr_web_search``.
    Blank tokens are dropped and a dimension with no non-blank values is treated as absent
    (``None`` → no filtering), so a malformed query never 500s the gallery. Values are passed
    through verbatim; unrecognised ones simply match nothing (see ``_apply_filters``).
    """

    def _single(key: str) -> str | None:
        raw = request.query_params.get(key)
        if raw is None:
            return None
        raw = raw.strip()
        return raw or None

    def _multi(key: str) -> list[str] | None:
        # Flatten repeated params and split each on commas, so ``?k=a,b``, ``?k=a&k=b`` and
        # any mix all yield the same list.
        values = [
            token.strip()
            for raw in request.query_params.getlist(key)
            for token in raw.split(",")
            if token.strip()
        ]
        return values or None

    return _single("name"), _multi("provider"), _multi("category")


def _apply_filters(
    items: list[dict[str, Any]],
    name: str | None,
    providers: list[str] | None,
    categories: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter gallery *items* by ``name`` (exact), ``provider`` and/or ``category``.

    ``provider`` and ``category`` are match-any **within** the dimension (an item is kept if
    its provider is one of *providers*, and/or if any of its ``categories`` is one of
    *categories* — each item carries both its leaf and its parent category, so a parent like
    ``dr_connectors`` matches every connector tool). The dimensions combine with **AND**.
    An unrecognised provider/category value simply matches no tools, so an unknown filter
    yields an empty page instead of a 500 — no separate known-value list is needed.
    """
    if name is not None:
        items = [item for item in items if item.get("name") == name]
    if providers is not None:
        wanted = set(providers)
        items = [item for item in items if item.get("provider") in wanted]
    if categories is not None:
        wanted = set(categories)
        items = [item for item in items if wanted.intersection(item.get("categories") or ())]
    return items


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
        ("/categories/", _categories_handler),
        ("/providers/", _providers_handler),
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

        # Filter before counting/paginating so totalCount/hasMore reflect the filtered set.
        name, providers, categories = _parse_filters(request)
        items = _apply_filters(items, name, providers, categories)

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


def _enum_items(mapping: dict[Any, str]) -> list[dict[str, str]]:
    """Serialise an ordered ``value -> label`` map into ``[{"value", "label"}]`` items.

    ``value`` is coerced to ``str`` so ``StrEnum`` keys (e.g. ``MCPToolCategory``) render as
    their plain ``dr_*`` strings — the exact values a tool item reports and the gallery's
    filter params accept.
    """
    return [{"value": str(value), "label": label} for value, label in mapping.items()]


async def _categories_handler(_request: Request) -> JSONResponse:
    """``GET /toolGallery/categories/`` — the tool-category filter enum (value + label)."""
    items = _enum_items(TOOL_CATEGORY_LABELS)
    return JSONResponse({"categories": items, "count": len(items)})


async def _providers_handler(_request: Request) -> JSONResponse:
    """``GET /toolGallery/providers/`` — the tool-provider filter enum (value + label)."""
    items = _enum_items(TOOL_PROVIDER_LABELS)
    return JSONResponse({"providers": items, "count": len(items)})


async def _gate_allows(gate: ToolGalleryGate, request: Request) -> bool:
    """Evaluate *gate*, treating any failure as "deny" (fail closed)."""
    try:
        return await gate(request)
    except Exception:
        logger.warning("toolGallery gate raised; denying access", exc_info=True)
        return False
