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

from collections.abc import Sequence

from fastmcp.experimental.transforms.code_mode import CodeMode
from fastmcp.server.transforms import GetToolNext
from fastmcp.tools import Tool
from fastmcp.utilities.versions import VersionSpec

from datarobot_genai.drtools.core.mode import MCPMode


class ConditionalCodeMode(CodeMode):
    """`CodeMode` that engages only when the request asks for `code_execute`.

    - Mode `tools` (default): catalog passes through unchanged; tools are
      listed and callable directly.
    - Mode `code_execute`: catalog collapses to CodeMode's discovery + execute
      meta-tools, and `get_tool` resolves only those names.
    """

    async def transform_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]:
        if MCPMode.from_current_http_request_headers() is MCPMode.CODE_EXECUTE:
            return await super().transform_tools(tools)
        return tools

    async def get_tool(
        self,
        name: str,
        call_next: GetToolNext,
        *,
        version: VersionSpec | None = None,
    ) -> Tool | None:
        if MCPMode.from_current_http_request_headers() == MCPMode.CODE_EXECUTE:
            return await super().get_tool(name, call_next, version=version)
        return await call_next(name, version=version)
