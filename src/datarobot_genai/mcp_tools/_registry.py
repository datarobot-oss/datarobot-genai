"""Tool registry with deduplication support.

Tools register themselves here. The MCP server reads from this registry
at startup. This allows tools to be used without MCP (e.g. injected
directly into LangGraph agents) per BUZZOK-29612.

Usage:
    from datarobot_genai.mcp_tools._registry import register_tool, get_all_tools

    # In a tool module:
    register_tool(
        name="my_tool",
        func=my_tool_func,
        description="Does a thing",
        category="wren_tools",
    )

    # At server startup:
    for name, tool_def in get_all_tools().items():
        mcp.tool(name=name)(tool_def.func)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name: str
    func: Callable
    description: str
    category: str  # "dr_tools", "wren_tools", "data_ops", etc.
    params_schema: dict = field(default_factory=dict)


_tool_registry: Dict[str, ToolDefinition] = {}


def register_tool(
    name: str,
    func: Callable,
    description: str,
    category: str,
    params_schema: dict | None = None,
) -> None:
    """Register a tool. If duplicate name exists, log warning and keep first."""
    if name in _tool_registry:
        logger.warning(
            "Duplicate tool '%s' from category '%s' "
            "— keeping existing from '%s'",
            name,
            category,
            _tool_registry[name].category,
        )
        return
    _tool_registry[name] = ToolDefinition(
        name=name,
        func=func,
        description=description,
        category=category,
        params_schema=params_schema or {},
    )


def get_all_tools() -> Dict[str, ToolDefinition]:
    """Return all registered tools."""
    return dict(_tool_registry)


def get_tools_by_category(category: str) -> Dict[str, ToolDefinition]:
    """Return tools filtered by category."""
    return {k: v for k, v in _tool_registry.items() if v.category == category}


def clear_registry() -> None:
    """Clear all registered tools. Used in tests."""
    _tool_registry.clear()
