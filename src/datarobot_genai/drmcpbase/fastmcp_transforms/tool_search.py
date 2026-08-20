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

"""Synthetic tools for the per-request ``search`` MCP mode.

With ``x-datarobot-mcp-mode: search`` the catalog collapses to two
synthetic tools (plus any allowlisted tools pinned by the client):

* ``tool_search`` — ranks the real catalog (as this request may see it:
  category gates and the tool allowlist still apply) against a natural
  language query and returns matching tool definitions.
* ``call_tool`` — proxy that executes a tool discovered via search, so a
  generic MCP client needs no re-listing loop to act on search results.

Ranking is pluggable via :class:`ToolSearchBackend` so a semantic backend
(e.g. a context-graph/embedding search) can replace the lexical default
without touching the mode mechanics.  Everything here is stateless per
request except the BM25 index, which is derived purely from the tool
catalog (rebuilt on catalog change, no user data).
"""

from collections.abc import Sequence
from typing import Annotated
from typing import Any
from typing import Protocol

from fastmcp.experimental.transforms.code_mode import GetToolCatalog
from fastmcp.experimental.transforms.code_mode import Search
from fastmcp.server.context import Context
from fastmcp.server.transforms.search.bm25 import BM25SearchTransform
from fastmcp.tools import Tool
from fastmcp.tools.base import ToolResult

TOOL_SEARCH_TOOL_NAME = "tool_search"
CALL_TOOL_PROXY_NAME = "call_tool"

DEFAULT_TOOL_SEARCH_MAX_RESULTS = 10


class ToolSearchBackend(Protocol):
    """Ranks *tools* against *query* and returns the matches, best first."""

    async def search(self, tools: Sequence[Tool], query: str) -> Sequence[Tool]: ...


class LexicalToolSearchBackend:
    """Default backend: FastMCP's self-contained BM25 index (no new deps).

    The wrapped transform never joins the pipeline — only its ``_search``
    ranking is reused, the same seam CodeMode's default discovery tool uses.
    """

    def __init__(self, max_results: int = DEFAULT_TOOL_SEARCH_MAX_RESULTS) -> None:
        self._bm25 = BM25SearchTransform(max_results=max_results)

    async def search(self, tools: Sequence[Tool], query: str) -> Sequence[Tool]:
        return await self._bm25._search(tools, query)


def build_tool_search_tool(
    get_catalog: GetToolCatalog,
    backend: ToolSearchBackend,
    *,
    max_results: int = DEFAULT_TOOL_SEARCH_MAX_RESULTS,
) -> Tool:
    """Build the synthetic ``tool_search`` tool over *get_catalog*.

    Results default to the ``detailed`` markdown rendering (parameter names
    and types) so the model can call a discovered tool through the
    ``call_tool`` proxy without another discovery round-trip.
    """
    factory = Search(
        search_fn=backend.search,
        name=TOOL_SEARCH_TOOL_NAME,
        default_detail="detailed",
        default_limit=max_results,
    )
    return factory(get_catalog)


def build_call_tool_proxy() -> Tool:
    """Build the synthetic ``call_tool`` proxy for search mode."""

    async def call_tool(
        name: Annotated[str, "The name of the tool to call"],
        arguments: Annotated[dict[str, Any] | None, "Arguments to pass to the tool"] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> ToolResult:
        """Call a tool by name with the given arguments.

        Use this to execute tools discovered via tool_search.
        """
        if name in {TOOL_SEARCH_TOOL_NAME, CALL_TOOL_PROXY_NAME}:
            raise ValueError(
                f"'{name}' is a synthetic search-mode tool and cannot be called "
                "via the call_tool proxy"
            )
        # Goes through the full get_tool pipeline, so category gates and the
        # tool allowlist apply to the proxied call exactly as to a direct one.
        return await ctx.fastmcp.call_tool(name, arguments)

    return Tool.from_function(fn=call_tool, name=CALL_TOOL_PROXY_NAME)
