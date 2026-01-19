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

"""LLM MCP Client implementations.

This module provides backwards compatibility by re-exporting all clients from the clients package.
New code should import directly from `datarobot_genai.drmcp.test_utils.clients`.
"""

# Re-export everything from clients package for backwards compatibility
from .clients import AnthropicMCPClient
from .clients import BaseLLMMCPClient
from .clients import BedrockMCPClient
from .clients import DRLLMGatewayMCPClient
from .clients import LLMResponse
from .clients import OpenAILLMMCPClient
from .clients import ToolCall

# Backwards compatibility alias
LLMMCPClient = OpenAILLMMCPClient

__all__ = [
    "AnthropicMCPClient",
    "BaseLLMMCPClient",
    "BedrockMCPClient",
    "DRLLMGatewayMCPClient",
    "LLMMCPClient",
    "LLMResponse",
    "OpenAILLMMCPClient",
    "ToolCall",
]
