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

"""Per-request gating of FastMCP's CodeMode transform.

CodeMode is a CatalogTransform that replaces the visible tool catalog with
discovery + execute meta-tools (`search`, `get_schema`, `execute`). It engages
globally if left unconditional. We want it engaged only when the request asks
for `code_execute` via the `x-datarobot-mcp-mode` header — see
`datarobot_genai.drtools.core.mode`.

The conditional check inside `transform_tools` / `get_tool` is safe: the
`CatalogTransform` base class uses an internal ContextVar to bypass itself on
re-entrant calls from `get_tool_catalog()`, so `super().transform_tools()` can
read the real catalog without recursing through this gate.
"""

from __future__ import annotations

from collections.abc import Sequence

from fastmcp.experimental.transforms.code_mode import CodeMode
from fastmcp.server.transforms import GetToolNext
from fastmcp.tools import Tool
from fastmcp.utilities.versions import VersionSpec

from datarobot_genai.drtools.core.mode import MCPMode
from datarobot_genai.drtools.core.mode import get_mcp_mode


class ConditionalCodeMode(CodeMode):
    """`CodeMode` that engages only when the request asks for `code_execute`.

    - Mode `tools` (default): catalog passes through unchanged; tools are
      listed and callable directly.
    - Mode `code_execute`: catalog collapses to CodeMode's discovery + execute
      meta-tools, and `get_tool` resolves only those names.
    """

    async def transform_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]:
        if get_mcp_mode() is MCPMode.CODE_EXECUTE:
            return await super().transform_tools(tools)
        return tools

    async def get_tool(
        self,
        name: str,
        call_next: GetToolNext,
        *,
        version: VersionSpec | None = None,
    ) -> Tool | None:
        if get_mcp_mode() is MCPMode.CODE_EXECUTE:
            return await super().get_tool(name, call_next, version=version)
        return await call_next(name, version=version)
