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

from collections.abc import Mapping
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Any

from fastmcp.server.dependencies import get_http_headers
from fastmcp.tools import Tool

MCP_MODE_HEADER = "x-datarobot-mcp-mode"
MCP_TOOLS_HEADER = "x-datarobot-mcp-tools"


def get_fast_mcp_http_headers(**kwargs: Any) -> dict[str, str]:
    return get_http_headers(include_all=True, **kwargs)


def get_header_value(headers: Mapping[str, str], name: str) -> str | None:
    """Read a header by lowercase name (FastMCP normalizes keys; scan only as fallback)."""
    value = headers.get(name)
    if value is not None:
        return value
    target = name.casefold()
    for key, header_value in headers.items():
        if key.casefold() == target:
            return header_value
    return None


def get_header_case_insensitive(headers: Mapping[str, str], name: str) -> str | None:
    return get_header_value(headers, name)


class MCPRequestMode(Enum):
    TOOLS = auto()
    CODE = auto()

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> "MCPRequestMode":
        raw = get_header_value(headers, MCP_MODE_HEADER) or ""
        token = raw.strip().upper()
        if not token:
            return cls.TOOLS
        try:
            return cls[token]
        except KeyError:
            return cls.TOOLS


def parse_tool_allowlist_header(raw: str | None) -> frozenset[str] | None:
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    entries = frozenset(part.strip() for part in stripped.split(",") if part.strip())
    if not entries:
        return None
    return entries


def filter_tools_by_allowlist(
    tools: Sequence[Tool],
    allowlist: frozenset[str],
) -> list[Tool]:
    return [tool for tool in tools if tool.name in allowlist]


def is_tool_name_allowed(name: str, allowlist: frozenset[str]) -> bool:
    return name in allowlist


@dataclass(frozen=True, slots=True)
class MCPRequestContext:
    mode: MCPRequestMode
    tool_allowlist: frozenset[str] | None

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> "MCPRequestContext":
        return cls(
            mode=MCPRequestMode.from_headers(headers),
            tool_allowlist=parse_tool_allowlist_header(get_header_value(headers, MCP_TOOLS_HEADER)),
        )

    @classmethod
    def from_current_http_request(cls) -> "MCPRequestContext":
        return get_request_context()


_request_context_cache: ContextVar[MCPRequestContext | None] = ContextVar(
    "_mcp_request_context_cache",
    default=None,
)


def get_request_context() -> MCPRequestContext:
    cached = _request_context_cache.get()
    if cached is not None:
        return cached
    ctx = MCPRequestContext.from_headers(get_fast_mcp_http_headers())
    _request_context_cache.set(ctx)
    return ctx
