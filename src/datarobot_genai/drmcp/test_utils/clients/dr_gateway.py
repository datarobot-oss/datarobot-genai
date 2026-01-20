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

"""DataRobot LLM Gateway MCP Client implementation."""

import openai

from .base import BaseLLMMCPClient


class DRLLMGatewayMCPClient(BaseLLMMCPClient):
    """
    Client for interacting with LLMs via MCP using DataRobot LLM Gateway.

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
                - datarobot_api_token: DataRobot API token
                - datarobot_endpoint: DataRobot endpoint URL (default: "https://app.datarobot.com/api/v2")
                - model: Model name (default: "gpt-4o-mini")
                - save_llm_responses: Whether to save responses (default: True)
        """
        super().__init__(config)

    def _create_llm_client(self, config_dict: dict) -> tuple[openai.OpenAI, str]:
        """Create the LLM client for DataRobot LLM Gateway."""
        datarobot_api_token = config_dict.get("datarobot_api_token")
        datarobot_endpoint = config_dict.get(
            "datarobot_endpoint", "https://app.datarobot.com/api/v2"
        )
        model = config_dict.get("model", "gpt-4o-mini")

        # Build gateway URL: {endpoint}/genai/llmgw
        gateway_url = datarobot_endpoint.rstrip("/") + "/genai/llmgw"

        client = openai.OpenAI(api_key=datarobot_api_token, base_url=gateway_url)
        return client, model
