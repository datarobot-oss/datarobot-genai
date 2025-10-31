# Copyright 2025 DataRobot, Inc.
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

import datetime
import json
import os
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from .openai_llm_mcp_client import LLMResponse

from dotenv import load_dotenv


def load_env() -> None:
    load_dotenv(dotenv_path=".env", verbose=True, override=True)


def format_tool_call(tool_call: dict[str, Any]) -> str:
    """Format a single tool call in a readable way."""
    return (
        f"Tool: {tool_call['tool_name']}\n"
        f"Parameters: {json.dumps(tool_call['parameters'], indent=2)}\n"
        f"Reasoning: {tool_call['reasoning']}"
    )


def format_response(response: "LLMResponse") -> str:
    """Format the LLM response in a readable way."""
    formatted_parts = []

    # Format the main content
    formatted_parts.append("=== LLM Response ===\n")
    formatted_parts.append(response.content)

    # Format tool calls if any
    if response.tool_calls:
        formatted_parts.append("\n=== Tools Used ===")
        for i, tool_call in enumerate(response.tool_calls, 1):
            formatted_parts.append(f"\nTool Call #{i}:")
            formatted_parts.append(
                format_tool_call(
                    {
                        "tool_name": tool_call.tool_name,
                        "parameters": tool_call.parameters,
                        "reasoning": tool_call.reasoning,
                    }
                )
            )

    # Format tool results if any
    if response.tool_results:
        formatted_parts.append("\n=== Tool Results ===")
        for i, result in enumerate(response.tool_results, 1):
            formatted_parts.append(f"\nResult #{i}:")
            formatted_parts.append(result)

    return "\n".join(formatted_parts)


def save_response_to_file(response: "LLMResponse", name: str | None = None) -> None:
    """Save the response to a file in a readable format.

    Args:
        response: The LLM response to save
        name: Optional name to use in the filename. If not provided,
              will use a timestamp only.
    """
    # Create responses directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    dir_path = "test_results/drmcp/.ete_responses/" + timestamp
    os.makedirs(dir_path, exist_ok=True)

    # Save both raw JSON and formatted text
    base_name = f"{name}" if name else "response"

    # Save formatted text
    with open(f"{dir_path}/{base_name}.txt", "w") as f:
        f.write(format_response(response))
