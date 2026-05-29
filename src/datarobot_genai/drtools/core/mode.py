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

"""MCP execution mode selection via the `x-datarobot-mcp-mode` request header.

The active mode is read from FastMCP's per-request HTTP context using the same
soft-import pattern as ``drtools/core/auth.py`` so this module remains usable
when fastmcp is not installed (tests, stdio transport, library consumers).
"""

from collections.abc import Mapping
from enum import Enum
from enum import auto
from typing import Any

try:
    from fastmcp.server.dependencies import get_http_headers

    def get_fast_mcp_http_headers(**kwargs: Any) -> dict[str, str]:
        # include_all=True so x-datarobot-* headers survive FastMCP's default exclusion list
        return get_http_headers(include_all=True, **kwargs)
except ImportError:

    def get_fast_mcp_http_headers(**kwargs: Any) -> dict[str, str]:
        return {}


def get_mcp_mode_header_key() -> str:
    return "x-datarobot-mcp-mode"


class MCPMode(Enum):
    TOOLS = auto()
    CODE_EXECUTE = auto()

    @classmethod
    def from_headers(cls, value: Mapping[str, str]) -> "MCPMode":
        try:
            return cls[value.get(get_mcp_mode_header_key(), "").upper()]
        except KeyError:
            return cls.TOOLS


def get_mcp_mode_from_headers() -> MCPMode:
    headers = get_fast_mcp_http_headers()
    return MCPMode.from_headers(headers)
