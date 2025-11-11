"""CrewAI utilities and helpers.

Public API:
- mcp_tools_context: Context manager returning available MCP tools for CrewAI.
- build_llm: Construct a CrewAI LLM configured for DataRobot endpoints.
- create_pipeline_interactions_from_messages: Convert messages to MultiTurnSample.
"""

from datarobot_genai.core.mcp.common import MCPConfig

from .agent import build_llm
from .agent import create_pipeline_interactions_from_messages
from .base import CrewAIAgent
from .events import CrewAIEventListener
from .mcp import mcp_tools_context

__all__ = [
    "mcp_tools_context",
    "CrewAIAgent",
    "build_llm",
    "create_pipeline_interactions_from_messages",
    "CrewAIEventListener",
    "MCPConfig",
]
