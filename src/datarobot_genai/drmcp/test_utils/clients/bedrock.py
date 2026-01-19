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

"""AWS Bedrock LLM MCP Client implementation (example).

This is an example implementation showing how easy it is to add a new LLM provider.
AWS Bedrock provides an OpenAI-compatible endpoint, so we can use the OpenAI SDK.
"""

import openai

from .base import BaseLLMMCPClient


class BedrockMCPClient(BaseLLMMCPClient):
    """
    Client for interacting with LLMs via MCP using AWS Bedrock.

    Note: Elicitation is handled at the protocol level by FastMCP's ctx.elicit().
    Tools using FastMCP's built-in elicitation will work automatically.

    Example:
        ```python
        config = {
            "aws_access_key_id": "AKIA...",
            "aws_secret_access_key": "...",
            "aws_region": "us-east-1",
            "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        }
        client = BedrockMCPClient(str(config))
        ```

    Note: For production use, consider using AWS IAM roles instead of access keys.
    """

    def __init__(
        self,
        config: str | dict,
    ):
        """
        Initialize the LLM MCP client.

        Args:
            config: Configuration string or dict with:
                - aws_access_key_id: AWS access key ID
                - aws_secret_access_key: AWS secret access key
                - aws_region: AWS region (default: "us-east-1")
                - model: Model name (default: "anthropic.claude-3-5-sonnet-20241022-v2:0")
                - save_llm_responses: Whether to save responses (default: True)
        """
        super().__init__(config)

    def _create_openai_client(self, config_dict: dict) -> tuple[openai.OpenAI, str]:
        """Create the OpenAI client for AWS Bedrock (OpenAI-compatible endpoint)."""
        aws_access_key_id = config_dict.get("aws_access_key_id")
        aws_secret_access_key = config_dict.get("aws_secret_access_key")
        aws_region = config_dict.get("aws_region", "us-east-1")
        model = config_dict.get("model", "anthropic.claude-3-5-sonnet-20241022-v2:0")

        # AWS Bedrock provides an OpenAI-compatible endpoint
        # Format: https://bedrock-runtime.{region}.amazonaws.com
        bedrock_endpoint = f"https://bedrock-runtime.{aws_region}.amazonaws.com"

        # For AWS authentication, we need to use AWS signature v4
        # The OpenAI SDK doesn't support this directly, so we'd need to use
        # boto3 or a custom auth handler. For this example, we'll use a placeholder.
        # In production, you'd want to use boto3's BedrockRuntimeClient or implement
        # AWS SigV4 signing for the OpenAI client.
        client = openai.OpenAI(
            api_key=f"{aws_access_key_id}:{aws_secret_access_key}",  # Placeholder
            base_url=bedrock_endpoint,
        )
        return client, model
