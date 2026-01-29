"""CrewAI utilities and helpers.

Public API:
- mcp_tools_context: Context manager returning available MCP tools for CrewAI.
- create_pipeline_interactions_from_messages: Convert messages to MultiTurnSample.
"""

from datarobot_genai.core.mcp.common import MCPConfig

from .agent import CrewAIAgent
from .agent import create_pipeline_interactions_from_messages
from .events import CrewAIEventListener
from .mcp import mcp_tools_context

__all__ = [
    "mcp_tools_context",
    "CrewAIAgent",
    "create_pipeline_interactions_from_messages",
    "CrewAIEventListener",
    "MCPConfig",
]
