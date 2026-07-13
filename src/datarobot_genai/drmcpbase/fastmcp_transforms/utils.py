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

from datarobot_genai.drmcpbase.dynamic_tools.enums import DataRobotMCPToolCategory
from datarobot_genai.drmcputils.categories import parse_tool_allowlist_header

MCP_MODE_HEADER = "x-datarobot-mcp-mode"
MCP_TOOLS_HEADER = "x-datarobot-mcp-tools"
# Extension point for user-defined toolsets (global-mcp only, resolved from mongo).
# Parsed here so the constant is co-located with the other header names.
MCP_TOOLSETS_HEADER = "x-datarobot-mcp-toolsets"

# Per-request category gates.  Optional booleans, default true; an explicit
# ``false`` disables the whole category for that request only.  Gates take
# precedence over the mode and the tools/toolsets allowlist: a tool in a
# disabled category stays hidden even when allowlisted.
MCP_ENABLE_PROXY_HEADER = "x-datarobot-mcp-enable-proxy"
MCP_ENABLE_DYNAMIC_TOOLS_HEADER = "x-datarobot-mcp-enable-dynamic-tools"

# Table-driven gate → category mapping so future gates (e.g. a generalized
# ``x-datarobot-mcp-disable=<category,...>``) only need a new row.  Values are
# ``DataRobotMCPToolCategory`` names exactly as the providers stamp them into
# ``tool.meta["tool_category"]``.
CATEGORY_GATE_HEADERS: Mapping[str, frozenset[str]] = {
    MCP_ENABLE_PROXY_HEADER: frozenset({DataRobotMCPToolCategory.PROXIED_USER_MCP.name}),
    MCP_ENABLE_DYNAMIC_TOOLS_HEADER: frozenset(
        {DataRobotMCPToolCategory.USER_TOOL_DEPLOYMENT.name}
    ),
}

_TRUE_HEADER_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_HEADER_VALUES = frozenset({"false", "0", "no", "off"})


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


def parse_bool_header(raw: str | None, *, default: bool = True) -> bool:
    """Parse an optional boolean header; unrecognized values fall back to *default*."""
    if raw is None:
        return default
    token = raw.strip().casefold()
    if token in _TRUE_HEADER_VALUES:
        return True
    if token in _FALSE_HEADER_VALUES:
        return False
    return default


def parse_disabled_categories(headers: Mapping[str, str]) -> frozenset[str]:
    """Resolve the category-gate headers to the set of disabled category names.

    Every gate defaults to enabled; only an explicit ``false`` disables its
    category for this request.
    """
    disabled: set[str] = set()
    for header_name, category_names in CATEGORY_GATE_HEADERS.items():
        if not parse_bool_header(get_header_value(headers, header_name)):
            disabled.update(category_names)
    return frozenset(disabled)


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


def get_tool_category(tool: Tool) -> str | None:
    """Category name stamped by the providers, or None for untagged (built-in) tools."""
    category = (tool.meta or {}).get("tool_category")
    return category if isinstance(category, str) else None


def is_tool_category_disabled(tool: Tool, disabled_categories: frozenset[str]) -> bool:
    if not disabled_categories:
        return False
    return get_tool_category(tool) in disabled_categories


def filter_tools_by_category_gates(
    tools: Sequence[Tool],
    disabled_categories: frozenset[str],
) -> Sequence[Tool]:
    if not disabled_categories:
        return tools
    return [tool for tool in tools if not is_tool_category_disabled(tool, disabled_categories)]


@dataclass(frozen=True, slots=True)
class MCPRequestContext:
    mode: MCPRequestMode
    tool_allowlist: frozenset[str] | None
    disabled_categories: frozenset[str] = frozenset()

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
            disabled_categories=parse_disabled_categories(headers),
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


def is_category_disabled_for_request(category_name: str) -> bool:
    """Return True when a category gate disables *category_name* for the current request.

    Safe to call from providers, including outside an HTTP request (startup
    retrospection, in-process clients): any failure to read the request context
    means "no gates", preserving the default-enabled behavior.
    """
    try:
        return category_name in get_request_context().disabled_categories
    except Exception:  # noqa: BLE001 — gates must never break the provider path
        return False
