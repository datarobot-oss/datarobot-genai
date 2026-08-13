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

"""``GET /metadata`` — the server's live tool/prompt/resource catalog for UIs.

user-mcp has long served an inline ``/metadata`` route (``drmcp/core/routes.py``)
whose ``config`` block is deeply tied to that server's ``MCPServerConfig``; that
route is intentionally left untouched. This registrar is the **shared** variant
for servers without one — global-mcp foremost — reusing the same catalog logic
(items with ``name`` + ``tags``, per-kind ``count``) behind the same fail-closed
gating as the other route groups.

Differences from the user-mcp route, by design:

- No ``config`` block by default. Global-mcp's configuration has nothing in
  common with ``MCPServerConfig``; a server that wants one injects a
  ``config_provider`` instead of this module guessing.
- Tool items also carry ``toolCategory`` — the provider's ``meta.tool_category``
  marker (``USER_TOOL_DEPLOYMENT`` for deployment tools, ``PROXIED_USER_MCP``
  for proxied user-MCP tools, ``None`` for static drtools tools) — so a UI can
  distinguish hosted tools from taxonomy tools without a second request.
"""

import logging
from collections.abc import Awaitable
from collections.abc import Callable
from http import HTTPStatus
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from datarobot_genai.drmcputils.routes.utils import CatalogProvider
from datarobot_genai.drmcputils.routes.utils import RouteGate
from datarobot_genai.drmcputils.routes.utils import register_gated_get
from datarobot_genai.drmcputils.routes.utils import resolve_catalog

logger = logging.getLogger(__name__)

# Matches the user-mcp route's path: bare ``/metadata``, no trailing slash.
METADATA_BASE_PATH = "/metadata"

# Supplies an optional server-specific ``config`` block per request. Async so a
# server can consult per-request context (feature flags, auth) when building it.
ConfigProvider = Callable[[Request], Awaitable[dict[str, Any]]]


def _tags(item: Any) -> list[str]:
    """Sorted tags of a FastMCP tool/prompt/resource (empty when untagged)."""
    return sorted(getattr(item, "tags", None) or [])


def _tool_category(tool: Any) -> str | None:
    """Read the provider's ``meta.tool_category`` marker (None for static drtools tools).

    Mirrors ``drmcputils.tool_gallery._tool_category``.
    """
    meta = getattr(tool, "meta", None) or {}
    return meta.get("tool_category")


def register_metadata_routes(
    mcp: Any,
    base_path: str = METADATA_BASE_PATH,
    gate: RouteGate | None = None,
    config_provider: ConfigProvider | None = None,
    catalog_provider: CatalogProvider | None = None,
) -> None:
    """Register ``GET <base_path>`` returning the live catalog summary.

    Args:
        mcp: FastMCP server instance.
        base_path: Route path (global-mcp uses the bare ``/metadata``; a mounted
            server would prefix it). Registered without a trailing slash to
            match the user-mcp contract.
        gate: Optional async ``(Request) -> bool``; ``False``/raise → ``404``.
            The route exposes tool names, so global-mcp reuses the gallery gate.
        config_provider: Optional async ``(Request) -> dict`` whose result is
            attached as the response's ``config`` key. Omitted → no ``config``.
        catalog_provider: Supplies the tool catalog to describe. Servers running
            the DataRobot catalog transform must pass
            ``drmcpbase.fastmcp_transforms.unfiltered_catalog_provider(mcp)`` so the
            caller's session headers cannot narrow it — see ``resolve_catalog``.
    """
    path = base_path.rstrip("/") or METADATA_BASE_PATH
    register_gated_get(
        mcp, path, _make_metadata_handler(mcp, config_provider, catalog_provider), gate
    )
    logger.info("metadata route registered at %s (gated=%s)", path, gate is not None)


def _make_metadata_handler(
    mcp: Any,
    config_provider: ConfigProvider | None,
    catalog_provider: CatalogProvider | None = None,
) -> Callable[[Request], Awaitable[JSONResponse]]:
    """Build the ``/metadata`` handler bound to *mcp*."""

    async def metadata_handler(request: Request) -> JSONResponse:
        try:
            # The catalog this server registers, not the slice the caller's session
            # headers would allow — see resolve_catalog for why the flag alone is not it.
            tools = await resolve_catalog(mcp, catalog_provider)
            prompts = await mcp.list_prompts(run_middleware=False)
            resources = await mcp.list_resources(run_middleware=False)

            tools_metadata = [
                {
                    "name": tool.name,
                    "tags": _tags(tool),
                    "toolCategory": _tool_category(tool),
                }
                for tool in tools
            ]
            prompts_metadata = [{"name": p.name, "tags": _tags(p)} for p in prompts]
            resources_metadata = [{"name": r.name, "tags": _tags(r)} for r in resources]

            content: dict[str, Any] = {
                "tools": {"items": tools_metadata, "count": len(tools_metadata)},
                "prompts": {"items": prompts_metadata, "count": len(prompts_metadata)},
                "resources": {"items": resources_metadata, "count": len(resources_metadata)},
            }
            if config_provider is not None:
                content["config"] = await config_provider(request)
            return JSONResponse(content)
        except Exception as exc:
            logger.exception("Failed to retrieve metadata")
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content={"error": f"Failed to retrieve metadata: {exc}"},
            )

    return metadata_handler
