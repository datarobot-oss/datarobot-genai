#!/usr/bin/env python3

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

"""Interactive MCP Client Test Script.

This script allows you to test arbitrary commands with the MCP server
using an LLM agent that can decide which tools to call.

Supports elicitation - when tools require user input (like authentication tokens),
the script will prompt you interactively.
"""

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult
from mcp.types import ElicitRequestParams
from mcp.types import ElicitResult
from mcp.types import ListToolsResult
from mcp.types import TextContent
from openai.types.chat.chat_completion import ChatCompletion

from datarobot_genai.drmcp import get_dr_mcp_server_url
from datarobot_genai.drmcp import get_headers
from datarobot_genai.drmcp.test_utils.utils import save_response_to_file


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
    """
    Client for interacting with LLMs via MCP.

    Note: Elicitation is handled at the protocol level by FastMCP's ctx.elicit().
    Tools using FastMCP's built-in elicitation will work automatically.
    """

    def __init__(
        self,
        config: str,
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
        """Add MCP tools to available tools list."""
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
        """
        Call an MCP tool and return the result as a string.

        Note: Elicitation is handled at the protocol level by FastMCP's ctx.elicit().
        Tools using FastMCP's built-in elicitation will work automatically.

        Args:
            tool_name: Name of the tool to call
            parameters: Parameters to pass to the tool
            mcp_session: MCP client session

        Returns
        -------
            Result text from the tool call
        """
        result: CallToolResult = await mcp_session.call_tool(tool_name, parameters)
        result_text = (
            result.content[0].text
            if result.content and isinstance(result.content[0], TextContent)
            else str(result.content)
        )

        # FastMCP handles elicitation at the protocol level via ctx.elicit()
        # No need to manually check for elicitation patterns
        return result_text

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
        """
        Process a prompt with MCP tool support and elicitation handling.

        This method:
        1. Adds MCP tools to available tools
        2. Sends prompt to LLM
        3. Processes tool calls
        4. Continues until LLM provides final response

        Note: Elicitation is handled at the protocol level by FastMCP's ctx.elicit().

        Args:
            prompt: User prompt
            mcp_session: MCP client session
            output_file_name: Optional file name to save response

        Returns
        -------
            LLMResponse with content, tool calls, and tool results
        """
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
                    "multiple tool calls or ask the user for more info, but prefer tool calls "
                    "when possible. "
                    "When dealing with file paths, use them as raw paths without converting "
                    "to file:// URLs."
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


async def test_mcp_interactive() -> None:
    """Test the MCP server interactively with LLM agent."""
    # Check for required environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("Please set it in your .env file or export it")
        return

    # Optional Azure OpenAI settings
    openai_api_base = os.environ.get("OPENAI_API_BASE")
    openai_api_deployment_id = os.environ.get("OPENAI_API_DEPLOYMENT_ID")
    openai_api_version = os.environ.get("OPENAI_API_VERSION")

    print("🤖 Initializing LLM MCP Client...")

    # Initialize the LLM client with elicitation handler
    config = {
        "openai_api_key": openai_api_key,
        "openai_api_base": openai_api_base,
        "openai_api_deployment_id": openai_api_deployment_id,
        "openai_api_version": openai_api_version,
        "save_llm_responses": False,
    }

    llm_client = LLMMCPClient(str(config))

    # Get MCP server URL
    mcp_server_url = get_dr_mcp_server_url()
    if not mcp_server_url:
        print("❌ Error: MCP server URL is not configured")
        print("Please set DR_MCP_SERVER_URL environment variable or run: task test-interactive")
        return

    print(f"🔗 Connecting to MCP server at: {mcp_server_url}")

    # Elicitation handler: prompt user for required values
    async def elicitation_handler(context, params: ElicitRequestParams) -> ElicitResult:
        print(f"\n📋 Elicitation Request: {params.message}")
        if params.requestedSchema:
            print(f"   Schema: {params.requestedSchema}")

        while True:
            try:
                response = input("   Enter value (or 'decline'/'cancel'): ").strip()
            except (EOFError, KeyboardInterrupt):
                return ElicitResult(action="cancel")

            if response.lower() == "decline":
                return ElicitResult(action="decline")
            if response.lower() == "cancel":
                return ElicitResult(action="cancel")
            if response:
                return ElicitResult(action="accept", content={"value": response})
            print("   Please enter a value or 'decline'/'cancel'")

    try:
        async with streamablehttp_client(
            url=mcp_server_url,
            headers=get_headers(),
        ) as (read_stream, write_stream, _):
            async with ClientSession(
                read_stream,
                write_stream,
                elicitation_callback=elicitation_handler,
            ) as session:
                await session.initialize()

                print("✅ Connected to MCP server!")
                print("📋 Available tools:")

                tools_result = await session.list_tools()
                for i, tool in enumerate(tools_result.tools, 1):
                    print(f"  {i}. {tool.name}: {tool.description}")

                print("\n" + "=" * 60)
                print("🎯 Interactive Testing Mode")
                print("=" * 60)
                print("Type your questions/commands. The AI will decide which tools to use.")
                print("If a tool requires additional information, you will be prompted.")
                print("Type 'quit' or 'exit' to stop.")
                print()

                while True:
                    try:
                        user_input = input("🤔 You: ").strip()

                        if user_input.lower() in ["quit", "exit", "q"]:
                            print("👋 Goodbye!")
                            break

                        if not user_input:
                            continue
                    except (EOFError, KeyboardInterrupt):
                        print("\n👋 Goodbye!")
                        break

                    print("🤖 AI is thinking...")

                    response = await llm_client.process_prompt_with_mcp_support(
                        prompt=user_input,
                        mcp_session=session,
                    )

                    print("\n🤖 AI Response:")
                    print("-" * 40)
                    print(response.content)

                    if response.tool_calls:
                        print("\n🔧 Tools Used:")
                        for i, tool_call in enumerate(response.tool_calls, 1):
                            print(f"  {i}. {tool_call.tool_name}")
                            print(f"     Parameters: {tool_call.parameters}")
                            print(f"     Reasoning: {tool_call.reasoning}")

                            if i <= len(response.tool_results):
                                result = response.tool_results[i - 1]
                                try:
                                    result_data = json.loads(result)
                                    if result_data.get("status") == "error":
                                        error_msg = result_data.get("error", "Unknown error")
                                        print(f"     ❌ Error: {error_msg}")
                                    elif result_data.get("status") == "success":
                                        print("     ✅ Success")
                                except json.JSONDecodeError:
                                    if len(result) > 100:
                                        print(f"     Result: {result[:100]}...")
                                    else:
                                        print(f"     Result: {result}")

                    print("\n" + "=" * 60)
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print(f"   Server URL: {mcp_server_url}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    # Ensure we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Please run this script from the project root")
        sys.exit(1)

    # Load environment variables from .env file
    print("📄 Loading environment variables...")
    load_dotenv()

    print("🚀 Starting Interactive MCP Client Test")
    print("Make sure the MCP server is running with: task drmcp-dev")
    print()

    asyncio.run(test_mcp_interactive())
