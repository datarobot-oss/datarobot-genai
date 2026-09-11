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

"""``/static/*`` — rich tool metadata routes for user-mcp UIs.

Routes in the group:
  - ``GET /static/tools/`` — the full paginated tool catalog.
  - ``GET /static/categories/`` — the category tree, with live per-node tool counts.
  - ``GET /static/providers/`` — the tool-provider filter enum (``value`` + ``label``).

Each ``tools/`` item carries ``required_scopes`` — the OAuth scopes a ``tools/call``
token must cover, combined across every declaration spelling and read with
``drmcpbase.oauth_scopes.declared_scopes_of_component``, the same function the
scope-validation middleware enforces with, so what is reported is what is enforced.

global-mcp and user-mcp both call ``register_static_routes`` from this module.
Tool Sets remain global-mcp-only at ``/toolGallery/toolSets/*``.
"""

import logging
from collections.abc import Awaitable
from collections.abc import Callable
from http import HTTPStatus
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from datarobot_genai.drmcpbase.oauth_scopes import declared_scopes_of_component
from datarobot_genai.drmcpbase.routes.helpers import apply_list_filters
from datarobot_genai.drmcpbase.routes.helpers import enum_items
from datarobot_genai.drmcpbase.routes.helpers import parse_list_filters
from datarobot_genai.drmcpbase.routes.helpers import parse_pagination
from datarobot_genai.drmcpbase.routes.utils import CatalogProvider
from datarobot_genai.drmcpbase.routes.utils import RouteGate
from datarobot_genai.drmcpbase.routes.utils import register_gated_get
from datarobot_genai.drmcpbase.routes.utils import resolve_catalog
from datarobot_genai.drmcputils.category_tree import build_category_tree
from datarobot_genai.drmcputils.tool_gallery import TOOL_PROVIDER_LABELS
from datarobot_genai.drmcputils.tool_gallery import build_tool_gallery_items
from datarobot_genai.drmcputils.tool_gallery import merge_tool_info

logger = logging.getLogger(__name__)

STATIC_BASE_PATH = "/static"
_DEFAULT_LIMIT = 100

StaticRouteGate = RouteGate
UiMetadataProvider = Callable[[], dict[str, dict[str, Any]]]


def register_static_routes(
    mcp: Any,
    base_path: str = STATIC_BASE_PATH,
    gate: StaticRouteGate | None = None,
    ui_metadata_provider: UiMetadataProvider | None = None,
    catalog_provider: CatalogProvider | None = None,
) -> None:
    """Register every ``/static/*`` discovery route on the FastMCP server, all sharing *gate*."""
    prefix = base_path.rstrip("/")

    routes: list[tuple[str, Callable[[Request], Awaitable[JSONResponse]]]] = [
        ("/tools/", _make_tools_handler(mcp, ui_metadata_provider, catalog_provider)),
        ("/categories/", _make_categories_handler(mcp, catalog_provider)),
        ("/providers/", _providers_handler),
    ]
    for sub_path, handler in routes:
        register_gated_get(mcp, f"{prefix}{sub_path}", handler, gate)

    logger.info(
        "static routes registered under %s (%d route(s), gated=%s)",
        prefix,
        len(routes),
        gate is not None,
    )


def _make_tools_handler(
    mcp: Any,
    ui_metadata_provider: UiMetadataProvider | None,
    catalog_provider: CatalogProvider | None = None,
) -> Callable[[Request], Awaitable[JSONResponse]]:
    async def tools_handler(request: Request) -> JSONResponse:
        try:
            tools = await resolve_catalog(mcp, catalog_provider)
        except Exception as exc:
            logger.exception("Failed to build the tool gallery")
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={"error": f"Failed to retrieve tool gallery: {exc}"},
            )
        ui_metadata = ui_metadata_provider() if ui_metadata_provider is not None else {}
        # ``required_scopes`` per item: every declaration spelling (required_scopes on
        # @dr_mcp_tool / @tool_metadata, MCP_OAUTH_TAG_SCOPES_<TAG>),
        # read with the same function the scope-validation middleware enforces with.
        merged = [
            merge_tool_info(tool, ui_metadata, declared_scopes_of_component) for tool in tools
        ]
        items = build_tool_gallery_items(merged)

        name, providers, categories = parse_list_filters(request)
        items = apply_list_filters(items, name, providers, categories)

        total_count = len(items)
        limit, offset = parse_pagination(request, default_limit=_DEFAULT_LIMIT)
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


def _make_categories_handler(
    mcp: Any, catalog_provider: CatalogProvider | None = None
) -> Callable[[Request], Awaitable[JSONResponse]]:
    async def categories_handler(_request: Request) -> JSONResponse:
        try:
            tools = await resolve_catalog(mcp, catalog_provider)
            categories, mapped = build_category_tree(tools)
        except Exception as exc:
            logger.exception("Failed to build tool categories")
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={"error": f"Failed to retrieve tool categories: {exc}"},
            )
        return JSONResponse(
            {"categories": categories, "count": len(categories), "totalCount": mapped}
        )

    return categories_handler


async def _providers_handler(_request: Request) -> JSONResponse:
    items = enum_items(TOOL_PROVIDER_LABELS)
    return JSONResponse({"providers": items, "count": len(items)})
