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

import logging
from collections.abc import Sequence
from typing import Any

from fastmcp.experimental.transforms.code_mode import CodeMode
from fastmcp.server.transforms import GetToolNext
from fastmcp.tools import Tool
from fastmcp.utilities.versions import VersionSpec

from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCPRequestContext
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCPRequestMode
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import filter_tools_by_allowlist
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import filter_tools_by_category_gates
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import get_request_context
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import is_tool_category_disabled
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import is_tool_name_allowed

logger = logging.getLogger(__name__)


class DataRobotMCPCatalogTransform(CodeMode):
    def _request_context(self) -> MCPRequestContext:
        return get_request_context()

    async def transform_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]:
        ctx = self._request_context()
        # Category gates run first — precedence: gates → mode → allowlist.  A tool
        # in a disabled category stays hidden even when allowlisted.
        tools = filter_tools_by_category_gates(tools, ctx.disabled_categories)
        if ctx.mode is MCPRequestMode.CODE_EXECUTE:
            return await super().transform_tools(tools)
        if ctx.tool_allowlist is None:
            return tools
        return filter_tools_by_allowlist(tools, ctx.tool_allowlist)

    async def get_tool(
        self,
        name: str,
        call_next: GetToolNext,
        *,
        version: VersionSpec | None = None,
    ) -> Tool | None:
        ctx = self._request_context()
        if ctx.mode is MCPRequestMode.CODE_EXECUTE:
            tool = await super().get_tool(name, call_next, version=version)
        else:
            if ctx.tool_allowlist is not None and not is_tool_name_allowed(
                name, ctx.tool_allowlist
            ):
                return None
            tool = await call_next(name, version=version)
        # Category gates apply in every mode: a tool in a disabled category is not
        # resolvable — and therefore not callable — for this request.  (CodeMode's
        # synthetic discovery tools carry no category meta and are never gated.)
        if tool is not None and is_tool_category_disabled(tool, ctx.disabled_categories):
            return None
        return tool


def register_mcp_catalog_transform(mcp: Any) -> None:
    mcp.add_transform(DataRobotMCPCatalogTransform())
    logger.info("DataRobot MCP catalog transform registered successfully")
