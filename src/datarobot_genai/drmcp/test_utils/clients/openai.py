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

"""OpenAI LLM MCP Client implementation."""

import openai

from .base import BaseLLMMCPClient


class OpenAILLMMCPClient(BaseLLMMCPClient):
    """
    Client for interacting with LLMs via MCP using OpenAI or Azure OpenAI.

    Note: Elicitation is handled at the protocol level by FastMCP's ctx.elicit().
    Tools using FastMCP's built-in elicitation will work automatically.
    """

    def __init__(
        self,
        config: str | dict,
    ):
        """
        Initialize the LLM MCP client.

        Args:
            config: Configuration string or dict with:
                - openai_api_key: OpenAI API key
                - openai_api_base: Optional Azure OpenAI endpoint
                - openai_api_deployment_id: Optional Azure deployment ID
                - openai_api_version: Optional Azure API version
                - model: Model name (default: "gpt-3.5-turbo")
                - save_llm_responses: Whether to save responses (default: True)
        """
        super().__init__(config)

    def _create_llm_client(
        self, config_dict: dict
    ) -> tuple[openai.OpenAI | openai.AzureOpenAI, str]:
        """Create the LLM client for OpenAI or Azure OpenAI."""
        openai_api_key = config_dict.get("openai_api_key")
        openai_api_base = config_dict.get("openai_api_base")
        openai_api_deployment_id = config_dict.get("openai_api_deployment_id")
        model = config_dict.get("model", "gpt-3.5-turbo")

        if openai_api_base and openai_api_deployment_id:
            # Azure OpenAI
            client = openai.AzureOpenAI(
                api_key=openai_api_key,
                azure_endpoint=openai_api_base,
                api_version=config_dict.get("openai_api_version", "2024-02-15-preview"),
            )
            return client, openai_api_deployment_id
        else:
            # Regular OpenAI
            client = openai.OpenAI(api_key=openai_api_key)  # type: ignore[assignment]
            return client, model
