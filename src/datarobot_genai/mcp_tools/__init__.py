"""MCP Tools subpackage for datarobot-genai.

This package provides tool implementations that can be used either:
- Via MCP server (registered at startup by drmcp)
- Directly injected into agent frameworks (LangGraph, CrewAI, etc.)

See BUZZOK-29612 for the decoupling rationale.

Importing this package triggers registration of all tools in all sub-packages.
"""
from . import data_ops, dr_tools, wren_tools  # noqa: F401 — triggers register_tool() calls
