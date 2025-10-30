from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from .agents.common import BaseAgent
from .agents.common import extract_user_prompt_content
from .agents.common import make_system_prompt
from .chat import CustomModelChatResponse
from .chat import CustomModelStreamingResponse
from .chat import ToolClient
from .chat import initialize_authorization_context
from .chat import to_custom_model_chat_response
from .chat import to_custom_model_streaming_response
from .cli import AgentEnvironment
from .cli import AgentKernel
from .utils.urls import get_api_base

__all__ = [
    "get_api_base",
    "CustomModelChatResponse",
    "CustomModelStreamingResponse",
    "ToolClient",
    "initialize_authorization_context",
    "to_custom_model_chat_response",
    "to_custom_model_streaming_response",
    "BaseAgent",
    "make_system_prompt",
    "extract_user_prompt_content",
    "AgentEnvironment",
    "AgentKernel",
    "__version__",
]

try:
    __version__ = version("datarobot-genai")
except PackageNotFoundError:  # pragma: no cover - during local dev without install
    __version__ = "0.0.0"
