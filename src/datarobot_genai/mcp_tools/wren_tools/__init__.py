"""Wren MCP tools ported to datarobot-genai.

Importing this package registers all wren tools in the mcp_tools registry.
These tools were migrated from wren-mcp with panel-specific types replaced
by plain dict/list returns.
"""
from . import (  # noqa: F401 — imports trigger register_tool() calls
    code_execution,
    data,
    deployment,
    model,
    optimization,
    search,
    use_case,
    vdb,
)
