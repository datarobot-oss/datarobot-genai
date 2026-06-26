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

from datarobot_genai.drmcpbase.fastmcp_transforms.categories import resolve_to_tool_names

MCP_MODE_HEADER = "x-datarobot-mcp-mode"
MCP_TOOLS_HEADER = "x-datarobot-mcp-tools"
# Extension point for user-defined toolsets (global-mcp only, resolved from mongo).
# Parsed here so the constant is co-located with the other header names.
MCP_TOOLSETS_HEADER = "x-datarobot-mcp-toolsets"


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
    CODE_EXECUTE = auto()

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


def _parse_header_entries(raw: str | None) -> frozenset[str] | None:
    """Split a comma-separated header value into a frozenset of stripped tokens.

    Returns None when the header is absent or blank (means "no filter").
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    entries = frozenset(part.strip() for part in stripped.split(",") if part.strip())
    return entries if entries else None


def parse_tool_allowlist_header(raw: str | None) -> frozenset[str] | None:
    """Parse the x-datarobot-mcp-tools header and resolve any category names.

    Category names (e.g. ``dr_connectors``, ``dr_connector_jira``) are expanded
    to the set of tool function names they contain.  Plain tool names and unknown
    entries are kept as-is.  Returns None when the header is absent or blank,
    meaning no tool filtering should be applied.
    """
    entries = _parse_header_entries(raw)
    if entries is None:
        return None
    return resolve_to_tool_names(entries)


def _resolve_toolsets(raw: str | None) -> frozenset[str]:
    """Stub: resolve x-datarobot-mcp-toolsets to tool names.

    User-defined toolsets are fetched from mongo by the global-mcp server and
    cached per request.  This stub always returns an empty set; the real
    implementation will be added in a future release for global-mcp only.
    """
    return frozenset()


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
        tools_allowlist = parse_tool_allowlist_header(get_header_value(headers, MCP_TOOLS_HEADER))
        toolsets_names = _resolve_toolsets(get_header_value(headers, MCP_TOOLSETS_HEADER))
        # Union the two resolved sets.  If either header is absent its result is
        # empty/None, so the union only matters when both headers are present.
        if tools_allowlist is not None and toolsets_names:
            combined: frozenset[str] | None = tools_allowlist | toolsets_names
        elif toolsets_names:
            combined = toolsets_names
        else:
            combined = tools_allowlist
        return cls(
            mode=MCPRequestMode.from_headers(headers),
            tool_allowlist=combined,
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
