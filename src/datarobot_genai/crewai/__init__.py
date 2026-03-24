"""CrewAI utilities and helpers.

Public API:
- mcp_tools_context: Context manager returning available MCP tools for CrewAI.
"""

from datarobot_genai.core.mcp.common import MCPConfig

from .agent import CrewAIAgent
from .mcp import mcp_tools_context
from .ragas_events import CrewAIRagasEventListener
from .streaming_events import CrewAIStreamingEventListener

__all__ = [
    "mcp_tools_context",
    "CrewAIAgent",
    "CrewAIRagasEventListener",
    "CrewAIStreamingEventListener",
    "MCPConfig",
]
