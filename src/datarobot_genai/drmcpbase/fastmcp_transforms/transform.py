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
from fastmcp.server.context import Context
from fastmcp.server.transforms import GetToolNext
from fastmcp.tools import Tool
from fastmcp.utilities.versions import VersionSpec

from datarobot_genai.drmcpbase.fastmcp_transforms.tool_search import LexicalToolSearchBackend
from datarobot_genai.drmcpbase.fastmcp_transforms.tool_search import ToolSearchBackend
from datarobot_genai.drmcpbase.fastmcp_transforms.tool_search import build_call_tool_proxy
from datarobot_genai.drmcpbase.fastmcp_transforms.tool_search import build_tool_search_tool
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCPRequestContext
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import MCPRequestMode
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import filter_tools_by_allowlist
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import filter_tools_by_category_gates
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import get_request_context
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import is_tool_category_disabled
from datarobot_genai.drmcpbase.fastmcp_transforms.utils import is_tool_name_allowed

logger = logging.getLogger(__name__)

_CODE_MODE_NOT_IMPLEMENTED_MSG = "Code mode is not implemented yet"


class DataRobotMCPCatalogTransform(CodeMode):
    """Per-request catalog shaping: category gates, tool allowlist, and modes.

    Enforcement contract — gates and the tool allowlist are hard caps applied
    in **every** mode, to listing, resolution (and therefore calling), and the
    catalog that the synthetic discovery/search tools read.  Modes only change
    presentation: ``tools`` lists the catalog directly, ``code``
    collapses it to discovery + execute meta-tools, ``search`` collapses it to
    ``tool_search`` + ``call_tool`` (plus any allowlisted tools, pinned).
    Synthetic mode-interface tools themselves are exempt from the caps — they
    are the mode's UI, not catalog tools.
    """

    def __init__(
        self,
        *,
        tool_search_backend: ToolSearchBackend | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._tool_search_backend = tool_search_backend or LexicalToolSearchBackend()
        self._built_search_mode_tools: list[Tool] | None = None

    def _request_context(self) -> MCPRequestContext:
        return get_request_context()

    def _build_search_mode_tools(self) -> list[Tool]:
        if self._built_search_mode_tools is None:
            self._built_search_mode_tools = [
                build_tool_search_tool(self.get_tool_catalog, self._tool_search_backend),
                build_call_tool_proxy(),
            ]
        return self._built_search_mode_tools

    async def get_tool_catalog(
        self, ctx: Context, *, run_middleware: bool = True
    ) -> Sequence[Tool]:
        """Fetch the real catalog *as this request may see it*.

        The base implementation bypasses this transform entirely, so the
        synthetic discovery/search/execute tools that read it would otherwise
        see (and leak) gated and non-allowlisted tools.  Re-apply both caps
        here — this is the single choke point for every synthetic-tool path.
        """
        tools = await super().get_tool_catalog(ctx, run_middleware=run_middleware)
        request_ctx = self._request_context()
        tools = filter_tools_by_category_gates(tools, request_ctx.disabled_categories)
        if request_ctx.tool_allowlist is not None:
            tools = filter_tools_by_allowlist(tools, request_ctx.tool_allowlist)
        return tools

    async def transform_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]:
        ctx = self._request_context()
        # Category gates run first — precedence: gates → mode → allowlist.  A tool
        # in a disabled category stays hidden even when allowlisted.
        tools = filter_tools_by_category_gates(tools, ctx.disabled_categories)
        if ctx.mode is MCPRequestMode.CODE:
            raise NotImplementedError(_CODE_MODE_NOT_IMPLEMENTED_MSG)
        if ctx.mode is MCPRequestMode.SEARCH:
            # Allowlisted tools stay pinned in the listing so a client that
            # re-lists with `x-datarobot-mcp-tools=<found names>` gets their
            # full definitions alongside the search interface.
            pinned = (
                filter_tools_by_allowlist(tools, ctx.tool_allowlist)
                if ctx.tool_allowlist is not None
                else []
            )
            return [*pinned, *self._build_search_mode_tools()]
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
        # Mode-interface (synthetic) tools resolve first: they carry no category
        # meta and are exempt from the allowlist — without them the mode itself
        # would be unusable under an allowlist.
        if ctx.mode is MCPRequestMode.SEARCH:
            for search_tool in self._build_search_mode_tools():
                if search_tool.name == name:
                    return search_tool
        elif ctx.mode is MCPRequestMode.CODE:
            raise NotImplementedError(_CODE_MODE_NOT_IMPLEMENTED_MSG)
        # Catalog tools: the allowlist is a hard cap in every mode.  (H5: the
        # code-mode path used to skip it, so switching the mode header made
        # every non-allowlisted tool resolvable and callable again.)
        if ctx.tool_allowlist is not None and not is_tool_name_allowed(name, ctx.tool_allowlist):
            return None
        tool = await call_next(name, version=version)
        # Category gates apply in every mode: a tool in a disabled category is not
        # resolvable — and therefore not callable — for this request.
        if tool is not None and is_tool_category_disabled(tool, ctx.disabled_categories):
            return None
        return tool


def register_mcp_catalog_transform(
    mcp: Any, *, tool_search_backend: ToolSearchBackend | None = None
) -> None:
    mcp.add_transform(DataRobotMCPCatalogTransform(tool_search_backend=tool_search_backend))
    logger.info("DataRobot MCP catalog transform registered successfully")
