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

"""Extended catalog transform with ``x-datarobot-mcp-toolsets`` resolution.

Subclasses ``DataRobotMCPCatalogTransform`` (defined in ``drmcpbase``) and
overrides ``_effective_allowlist`` to resolve tool set names from MongoDB.

This lives in ``drmcp`` (not ``drmcpbase``) so it can import motor/MongoDB
code without violating the import boundary: drmcpbase cannot import drmcp.

Tech spec §5.1 — §5.5.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from fastmcp.server.transforms import GetToolNext
from fastmcp.tools import Tool
from fastmcp.utilities.versions import VersionSpec

from datarobot_genai.drmcpbase.fastmcp_transforms.transform import DataRobotMCPCatalogTransform
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCPRequestContext
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCPRequestMode
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import filter_tools_by_allowlist
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import get_request_context
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import is_tool_name_allowed
from datarobot_genai.drmcp.core.auth_utils import get_created_by

logger = logging.getLogger(__name__)


async def _resolve_tool_set_names_to_allowlist(
    created_by: str,
    names: list[str],
) -> frozenset[str]:
    """Resolve tool set names to a flat frozenset of tool names via MongoDB.

    Unknown set names contribute nothing (silently ignored per spec §5.6).
    Mongo errors fail-open: log the exception and return an empty set.
    """
    if not names:
        return frozenset()
    try:
        from datarobot_genai.drmcp.core.platform.mongodb import get_db
        from datarobot_genai.drmcp.core.tool_sets import ToolSetCRUDService

        db = await get_db()
        service = ToolSetCRUDService(db)
        tool_sets = await service.find_by_names(created_by=created_by, names=names)
        result: set[str] = set()
        for ts in tool_sets:
            result.update(ts.tool_names)
        return frozenset(result)
    except Exception:
        logger.exception(
            "Failed to resolve tool sets %s for user %s — failing open.", names, created_by
        )
        return frozenset()


class ToolGalleryCatalogTransform(DataRobotMCPCatalogTransform):
    """Catalog transform that resolves ``x-datarobot-mcp-toolsets`` from MongoDB.

    Extends the base transform with tool set resolution (§5).  All existing
    behaviour (``x-datarobot-mcp-tools``, ``code_execute`` mode bypass) is
    preserved unchanged.
    """

    async def _effective_allowlist(
        self, ctx: MCPRequestContext
    ) -> frozenset[str] | None:
        """Compute the combined allowlist from both headers.

        Precedence (§5.6):
        - Neither header → None (no filtering).
        - Only x-datarobot-mcp-tools → existing allowlist unchanged.
        - x-datarobot-mcp-toolsets present → resolve to tool names from DB;
          union with x-datarobot-mcp-tools if also present.
        """
        tool_set_names: list[str] = getattr(ctx, "tool_set_names", [])
        explicit_allowlist: frozenset[str] | None = ctx.tool_allowlist

        if not tool_set_names:
            return explicit_allowlist

        created_by = get_created_by()
        if not created_by:
            logger.warning(
                "x-datarobot-mcp-toolsets present but no caller identity available; "
                "falling back to explicit allowlist only."
            )
            return explicit_allowlist

        resolved = await _resolve_tool_set_names_to_allowlist(created_by, tool_set_names)

        if explicit_allowlist is not None:
            return explicit_allowlist | resolved
        return resolved if resolved else frozenset()

    async def transform_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]:
        ctx = get_request_context()
        if ctx.mode is MCPRequestMode.CODE_EXECUTE:
            return await super(DataRobotMCPCatalogTransform, self).transform_tools(tools)
        allowlist = await self._effective_allowlist(ctx)
        if allowlist is None:
            return tools
        return filter_tools_by_allowlist(tools, allowlist)

    async def get_tool(
        self,
        name: str,
        call_next: GetToolNext,
        *,
        version: VersionSpec | None = None,
    ) -> Tool | None:
        ctx = get_request_context()
        if ctx.mode is MCPRequestMode.CODE_EXECUTE:
            return await super(DataRobotMCPCatalogTransform, self).get_tool(
                name, call_next, version=version
            )
        allowlist = await self._effective_allowlist(ctx)
        if allowlist is not None and not is_tool_name_allowed(name, allowlist):
            return None
        return await call_next(name, version=version)


def register_tool_gallery_transform(mcp: Any) -> None:
    """Register the extended catalog transform on *mcp*."""
    mcp.add_transform(ToolGalleryCatalogTransform())
    logger.info("ToolGalleryCatalogTransform registered (tool sets + explicit allowlist).")
