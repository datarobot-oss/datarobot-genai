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

"""Anthropic LLM MCP Client implementation (example).

This is an example implementation showing how easy it is to add a new LLM provider.
Anthropic's API is OpenAI-compatible, so we can use the OpenAI SDK with their endpoint.
"""

import openai

from .base import BaseLLMMCPClient


class AnthropicMCPClient(BaseLLMMCPClient):
    """
    Client for interacting with LLMs via MCP using Anthropic Claude.

    Note: Elicitation is handled at the protocol level by FastMCP's ctx.elicit().
    Tools using FastMCP's built-in elicitation will work automatically.

    Example:
        ```python
        config = {
            "anthropic_api_key": "sk-ant-...",
            "model": "claude-3-5-sonnet-20241022",
        }
        client = AnthropicMCPClient(str(config))
        ```
    """

    def __init__(
        self,
        config: str | dict,
    ):
        """
        Initialize the LLM MCP client.

        Args:
            config: Configuration string or dict with:
                - anthropic_api_key: Anthropic API key
                - model: Model name (default: "claude-3-5-sonnet-20241022")
                - save_llm_responses: Whether to save responses (default: True)
        """
        super().__init__(config)

    def _create_llm_client(self, config_dict: dict) -> tuple[openai.OpenAI, str]:
        """Create the LLM client for Anthropic (OpenAI-compatible endpoint)."""
        anthropic_api_key = config_dict.get("anthropic_api_key")
        model = config_dict.get("model", "claude-3-5-sonnet-20241022")

        # Anthropic provides an OpenAI-compatible endpoint
        client = openai.OpenAI(
            api_key=anthropic_api_key,
            base_url="https://api.anthropic.com/v1",
        )
        return client, model
