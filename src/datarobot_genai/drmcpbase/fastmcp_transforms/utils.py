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

from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from enum import auto
from logging import getLogger
from typing import Any

from fastmcp.server.dependencies import get_http_headers
from fastmcp.tools import Tool

from datarobot_genai.drmcpbase.dynamic_tools.enums import DataRobotMCPToolCategory
from datarobot_genai.drmcputils.categories import ToolAllowlist
from datarobot_genai.drmcputils.categories import parse_tool_allowlist_header
from datarobot_genai.drmcputils.categories import parse_toolset_names_header
from datarobot_genai.drmcputils.tool_gallery import marked_kind

logger = getLogger(__name__)

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
    # Collapse the catalog to discovery + execute meta-tools (CodeMode).
    CODE = auto()
    # Collapse the catalog to a `tool_search` + `call_tool` pair for
    # just-in-time tool discovery; see fastmcp_transforms/tool_search.py.
    SEARCH = auto()

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


# Expand ``x-datarobot-mcp-toolsets`` bundle names to concrete tool names (MCP allowlist).
# Global-mcp registers an async mongo-backed expander at startup; default is no-op.
ToolsetsAllowlistExpander = Callable[[frozenset[str]], Awaitable[frozenset[str]]]

# Single-element list avoids a `global` statement (PLW0603).
_toolsets_allowlist_expander: list[ToolsetsAllowlistExpander | None] = [None]


def register_toolsets_allowlist_expander(expander: ToolsetsAllowlistExpander | None) -> None:
    """Install async expansion of toolset bundle names → tool function names.

    Global-mcp registers a mongo-backed expander once at startup. User-mcp leaves the
    default, where a toolsets header resolves to no tools at all — see
    :func:`effective_tool_allowlist` for what that means for the request. Pass ``None`` to
    reset (tests).
    """
    _toolsets_allowlist_expander[0] = expander
    # Swapping the expander invalidates anything the previous one produced. In production
    # this runs once at startup, before any request; in tests it is what keeps one case's
    # expansion from being served to the next.
    _toolset_expansion_cache.set(None)


async def expand_toolset_names_to_tools(names: frozenset[str] | None) -> frozenset[str]:
    """Expand parsed toolset bundle *names* to tool function names for allowlisting.

    Every failure resolves to the empty set, which :func:`effective_tool_allowlist` keeps
    as an *empty allowlist* rather than "no allowlist" — degrading toward fewer tools, not
    more.
    """
    expander = _toolsets_allowlist_expander[0]
    if not names:
        return frozenset()
    if expander is None:
        logger.warning(
            "Received %s but this server has no Tool Sets expander registered "
            "(Tool Sets are global-mcp only); the request is allowed no tools.",
            MCP_TOOLSETS_HEADER,
        )
        return frozenset()
    try:
        return await expander(names)
    except Exception:
        logger.warning(
            "Tool Sets expander failed; the request is allowed no tools from %s",
            MCP_TOOLSETS_HEADER,
            exc_info=True,
        )
        return frozenset()


def is_tool_allowed(tool: Tool, allowlist: ToolAllowlist) -> bool:
    """Whether *tool* survives *allowlist*, taking into account how it was named.

    Three ways in, and which apply depends on what kind of tool this is:

    - an **explicit** name admits anything — the client asked for this tool by name;
    - a **derived** name (from expanding a static category) admits only *built-ins*,
      because the static taxonomy describes DataRobot's own tools. A user-authored
      tool that merely shares a name is not what ``dr_db`` meant;
    - a **bucket** admits a tool whose marker puts it there.

    Ordered so the common paths never read the tool's marker: an explicit hit returns
    at once, and a name in neither set with no bucket requested is rejected before the
    marker is touched. Only a name that *could* match — or a request that named a
    bucket — pays for the lookup.
    """
    name = tool.name
    if name in allowlist.explicit:
        return True
    in_derived = name in allowlist.derived
    if not in_derived and not allowlist.buckets:
        return False
    kind = marked_kind(get_tool_category(tool))
    if kind is None:
        # A built-in: the static taxonomy speaks for it.
        return in_derived
    # Marker-classified (user / deployment / proxied). Its category comes from the
    # marker, so only a bucket can admit it — never a name a static category expanded
    # to. Proxied tools carry a marker but no category, so no bucket names them.
    bucket = kind["category"]
    return bucket is not None and bucket in allowlist.buckets


def filter_tools_by_allowlist(
    tools: Sequence[Tool],
    allowlist: ToolAllowlist,
) -> list[Tool]:
    """Apply *allowlist* to a whole catalog.

    The fast path is not a second copy of the rule — it is the one case where the rule
    provably collapses. With no derived names and no buckets, :func:`is_tool_allowed`
    can only return ``name in explicit``, so the marker is never consulted and this is
    the same single set lookup per tool the allowlist has always cost.
    """
    if not allowlist.derived and not allowlist.buckets:
        explicit = allowlist.explicit
        return [tool for tool in tools if tool.name in explicit]
    return [tool for tool in tools if is_tool_allowed(tool, allowlist)]


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
    tool_allowlist: ToolAllowlist | None
    disabled_categories: frozenset[str] = frozenset()
    # Raw bundle names from ``x-datarobot-mcp-toolsets`` — expanded asynchronously.
    toolset_names: frozenset[str] | None = None

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> "MCPRequestContext":
        tools_allowlist = parse_tool_allowlist_header(get_header_value(headers, MCP_TOOLS_HEADER))
        toolset_names = parse_toolset_names_header(get_header_value(headers, MCP_TOOLSETS_HEADER))
        return cls(
            mode=MCPRequestMode.from_headers(headers),
            tool_allowlist=tools_allowlist,
            disabled_categories=parse_disabled_categories(headers),
            toolset_names=toolset_names,
        )

    @classmethod
    def from_current_http_request(cls) -> "MCPRequestContext":
        return get_request_context()


async def effective_tool_allowlist(ctx: MCPRequestContext) -> ToolAllowlist | None:
    """Union ``x-datarobot-mcp-tools`` and expanded ``x-datarobot-mcp-toolsets`` names.

    ``None`` means *no allowlist* — every tool is permitted — so it is only ever returned
    when neither header narrowed the request. An empty ``frozenset`` is the opposite
    answer: the caller asked to be narrowed and nothing survived, so nothing is permitted.

    Collapsing those two is the failure this guards against. ``x-datarobot-mcp-toolsets``
    expands to nothing whenever Mongo is unreachable, the Tool Sets feature is off, this
    server has no expander, or the named set was deleted — and if an empty expansion fell
    through to ``ctx.tool_allowlist`` (``None`` when the toolsets header was the *only*
    filter sent), a session that asked for one bundle would silently receive the entire
    catalog. Precedence is unchanged: an allowlist is still a hard cap in every mode.
    """
    if not ctx.toolset_names:
        return ctx.tool_allowlist
    expanded = await _expand_toolset_names_cached(ctx)
    if ctx.tool_allowlist is None:
        return ToolAllowlist(explicit=expanded)
    return ctx.tool_allowlist.with_explicit(expanded)


# Cache of one (context, expansion) pair, valid for the current request only — the same
# lifetime and isolation as _request_context_cache below. The expander reaches Mongo in
# global-mcp and is consulted on every tools/list *and* every tools/call, several times per
# request once get_tool_catalog, transform_tools and get_tool each ask for the allowlist.
_toolset_expansion_cache: ContextVar[tuple[MCPRequestContext, frozenset[str]] | None] = ContextVar(
    "_mcp_toolset_expansion_cache",
    default=None,
)


async def _expand_toolset_names_cached(ctx: MCPRequestContext) -> frozenset[str]:
    """Expand ``ctx.toolset_names`` once per request context.

    Keyed on the context itself rather than just stored, so a caller passing a different
    context (tests, in-process clients) never reads another context's answer.
    """
    cached = _toolset_expansion_cache.get()
    if cached is not None and cached[0] == ctx:
        return cached[1]
    expanded = await expand_toolset_names_to_tools(ctx.toolset_names)
    _toolset_expansion_cache.set((ctx, expanded))
    return expanded


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


# What the catalog transform sees when the session filter is neutralized: default mode, no
# allowlist, no gates, no toolsets — i.e. every filtering branch of the transform is off.
_UNFILTERED_REQUEST_CONTEXT = MCPRequestContext(mode=MCPRequestMode.TOOLS, tool_allowlist=None)


def unfiltered_catalog_provider(mcp: Any) -> Callable[[], Awaitable[Sequence[Tool]]]:
    """Build a catalog provider that ignores the caller's ``x-datarobot-mcp-*`` headers.

    For the describe-the-server REST routes (``/toolGallery/tools/``,
    ``/toolGallery/categories/``,
    ``/metadata``), which report what the server registers rather than what the current
    request may call. ``list_tools(run_middleware=False)`` is not enough on its own —
    FastMCP still applies registered catalog transforms, and
    :class:`DataRobotMCPCatalogTransform` is exactly where the session filter is enforced.

    Pre-seeding the per-request context cache with a neutral context is what turns the
    transform into a pass-through, because every branch of it reads the context from there.
    The token is reset on the way out, so an MCP call sharing the same context (there is
    none today, but the routes and the protocol do share a process) still resolves its own
    headers afterwards.
    """

    async def provider() -> Sequence[Tool]:
        token = _request_context_cache.set(_UNFILTERED_REQUEST_CONTEXT)
        try:
            return await mcp.list_tools(run_middleware=False)
        finally:
            _request_context_cache.reset(token)

    return provider


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
