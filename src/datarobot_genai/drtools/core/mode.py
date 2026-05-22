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

from __future__ import annotations

from enum import Enum
from enum import auto
from typing import Any

try:
    from fastmcp.server.dependencies import get_http_headers as _fm_get_http_headers

    def _get_http_headers(**kwargs: Any) -> dict[str, str]:
        # include_all=True so x-datarobot-* headers survive FastMCP's default exclusion list
        return _fm_get_http_headers(include_all=True, **kwargs)
except ImportError:

    def _get_http_headers(**kwargs: Any) -> dict[str, str]:
        return {}


MCP_MODE_HEADER = "x-datarobot-mcp-mode"


class MCPMode(Enum):
    """Execution mode selected by the `x-datarobot-mcp-mode` request header."""

    TOOLS = auto()
    CODE_EXECUTE = auto()

    @classmethod
    def from_header(cls, value: str) -> MCPMode:
        """Parse the header string. Unknown / empty values fall back to TOOLS."""
        _map = {
            "tools": cls.TOOLS,
            "code_execute": cls.CODE_EXECUTE,
        }
        return _map.get(value.strip().lower(), cls.TOOLS)

    def to_header(self) -> str:
        """Serialise to the header value sent by callers."""
        return self.name.lower()


def get_mcp_mode() -> MCPMode:
    """Return the active MCPMode for the current request.

    Reads ``x-datarobot-mcp-mode`` from FastMCP's per-request HTTP headers.
    Defaults to ``MCPMode.TOOLS`` when no header is present or when no request
    context is active (startup, stdio transport, unit tests without fastmcp).
    """
    headers = _get_http_headers()
    return MCPMode.from_header(headers.get(MCP_MODE_HEADER, ""))
