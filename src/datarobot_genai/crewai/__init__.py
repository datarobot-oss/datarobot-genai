"""CrewAI utilities and helpers.

Public API:
- mcp_tools_context: Context manager returning available MCP tools for CrewAI.
"""

from datarobot_genai.core.mcp.common import MCPConfig

from .agent import CrewAIAgent
from .events import CrewAIRagasEventListener
from .mcp import mcp_tools_context

__all__ = [
    "mcp_tools_context",
    "CrewAIAgent",
    "CrewAIRagasEventListener",
    "MCPConfig",
]
