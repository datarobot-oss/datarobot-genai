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
from collections.abc import Mapping
from collections.abc import Sequence
from enum import Enum
from enum import auto
from typing import Any

from fastmcp.experimental.transforms.code_mode import CodeMode
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.transforms import GetToolNext
from fastmcp.tools import Tool
from fastmcp.utilities.versions import VersionSpec


def get_fast_mcp_http_headers(**kwargs: Any) -> dict[str, str]:
    # include_all=True so x-datarobot-* headers survive FastMCP's default exclusion list
    return get_http_headers(include_all=True, **kwargs)


class MCPMode(Enum):
    TOOLS = auto()
    CODE_EXECUTE = auto()

    @classmethod
    def from_current_http_request_headers(cls) -> "MCPMode":
        headers = get_fast_mcp_http_headers()
        return cls._from_headers(headers)

    @staticmethod
    def _get_mcp_mode_header_key() -> str:
        return "x-datarobot-mcp-mode"

    @classmethod
    def _from_headers(cls, value: Mapping[str, str]) -> "MCPMode":
        try:
            return cls[value.get(cls._get_mcp_mode_header_key(), "").upper()]
        except KeyError:
            return cls.TOOLS


logger = logging.getLogger(__name__)


class ConditionalCodeMode(CodeMode):
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


def initialize_conditional_code_mode_transform(mcp: Any) -> None:
    mcp.add_transform(ConditionalCodeMode())
    logger.info("Code mode transform registered successfully")
