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

import json
from typing import Any

import openai
from mcp import ClientSession
from mcp.types import CallToolResult
from mcp.types import ListToolsResult
from mcp.types import TextContent
from openai.types.chat.chat_completion import ChatCompletion

from .utils import save_response_to_file


class ToolCall:
    """Represents a tool call with its parameters and reasoning."""

    def __init__(self, tool_name: str, parameters: dict[str, Any], reasoning: str):
        self.tool_name = tool_name
        self.parameters = parameters
        self.reasoning = reasoning


class LLMResponse:
    """Represents an LLM response with content and tool calls."""

    def __init__(self, content: str, tool_calls: list[ToolCall], tool_results: list[str]):
        self.content = content
        self.tool_calls = tool_calls
        self.tool_results = tool_results


class LLMMCPClient:
    """Client for interacting with LLMs via MCP."""

    def __init__(self, config: str):
        """Initialize the LLM MCP client."""
        # Parse config string to extract parameters
        config_dict = eval(config) if isinstance(config, str) else config

        openai_api_key = config_dict.get("openai_api_key")
        openai_api_base = config_dict.get("openai_api_base")
        openai_api_deployment_id = config_dict.get("openai_api_deployment_id")
        model = config_dict.get("model", "gpt-3.5-turbo")
        save_llm_responses = config_dict.get("save_llm_responses", True)

        if openai_api_base and openai_api_deployment_id:
            # Azure OpenAI
            self.openai_client = openai.AzureOpenAI(
                api_key=openai_api_key,
                azure_endpoint=openai_api_base,
                api_version=config_dict.get("openai_api_version", "2024-02-15-preview"),
            )
            self.model = openai_api_deployment_id
        else:
            # Regular OpenAI
            self.openai_client = openai.OpenAI(api_key=openai_api_key)  # type: ignore[assignment]
            self.model = model

        self.save_llm_responses = save_llm_responses
        self.available_tools: list[dict[str, Any]] = []
        self.available_prompts: list[dict[str, Any]] = []
        self.available_resources: list[dict[str, Any]] = []

    async def _add_mcp_tool_to_available_tools(self, mcp_session: ClientSession) -> None:
        """Add a tool to the available tools."""
        tools_result: ListToolsResult = await mcp_session.list_tools()
        self.available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]

    async def _call_mcp_tool(
        self, tool_name: str, parameters: dict[str, Any], mcp_session: ClientSession
    ) -> str:
        """Call an MCP tool and return the result as a string."""
        result: CallToolResult = await mcp_session.call_tool(tool_name, parameters)
        content = (
            result.content[0].text
            if result.content and isinstance(result.content[0], TextContent)
            else str(result.content)
        )
        if result.structuredContent is not None:
            structured_content = json.dumps(result.structuredContent)
        else:
            structured_content = ""
        return f"Content: {content}\nStructured content: {structured_content}"

    async def _process_tool_calls(
        self,
        response: ChatCompletion,
        messages: list[Any],
        mcp_session: ClientSession,
    ) -> tuple[list[ToolCall], list[str]]:
        """Process tool calls from the response, and return the tool calls and tool results."""
        tool_calls = []
        tool_results = []

        # If the response has tool calls, process them
        if response.choices[0].message.tool_calls:
            messages.append(response.choices[0].message)  # Add assistant's message with tool calls

            for tool_call in response.choices[0].message.tool_calls:
                tool_name = tool_call.function.name  # type: ignore[union-attr]
                parameters = json.loads(tool_call.function.arguments)  # type: ignore[union-attr]

                tool_calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        parameters=parameters,
                        reasoning="Tool selected by LLM",
                    )
                )

                try:
                    result_text = await self._call_mcp_tool(tool_name, parameters, mcp_session)
                    tool_results.append(result_text)

                    # Add tool result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "content": result_text,
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                        }
                    )
                except Exception as e:
                    error_msg = f"Error calling {tool_name}: {str(e)}"
                    tool_results.append(error_msg)
                    messages.append(
                        {
                            "role": "tool",
                            "content": error_msg,
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                        }
                    )

        return tool_calls, tool_results

    async def _get_llm_response(
        self, messages: list[dict[str, Any]], allow_tool_calls: bool = True
    ) -> Any:
        """Get a response from the LLM with optional tool calling capability."""
        kwargs = {
            "model": self.model,
            "messages": messages,
        }

        if allow_tool_calls and self.available_tools:
            kwargs["tools"] = self.available_tools
            kwargs["tool_choice"] = "auto"

        return self.openai_client.chat.completions.create(**kwargs)

    async def process_prompt_with_mcp_support(
        self, prompt: str, mcp_session: ClientSession, output_file_name: str = ""
    ) -> LLMResponse:
        """Process a prompt with MCP tool support."""
        # Add MCP tools to available tools
        await self._add_mcp_tool_to_available_tools(mcp_session)

        if output_file_name:
            print(f"Processing prompt for test: {output_file_name}")

        # Initialize conversation
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant that can use tools to help users. "
                    "If you need more information to provide a complete response, you can make "
                    "multiple tool calls. When dealing with file paths, use them as raw paths "
                    "without converting to file:// URLs."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        all_tool_calls = []
        all_tool_results = []

        while True:
            # Get LLM response
            response = await self._get_llm_response(messages)

            # If no tool calls in response, this is the final response
            if not response.choices[0].message.tool_calls:
                final_response = response.choices[0].message.content
                break

            # Process tool calls
            tool_calls, tool_results = await self._process_tool_calls(
                response, messages, mcp_session
            )
            all_tool_calls.extend(tool_calls)
            all_tool_results.extend(tool_results)

            # Get another LLM response to see if we need more tool calls
            response = await self._get_llm_response(messages, allow_tool_calls=True)

            # If no more tool calls needed, this is the final response
            if not response.choices[0].message.tool_calls:
                final_response = response.choices[0].message.content
                break

        clean_content = final_response.replace("*", "").lower()

        llm_response = LLMResponse(
            content=clean_content,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
        )

        if self.save_llm_responses:
            save_response_to_file(llm_response, name=output_file_name)

        return llm_response
