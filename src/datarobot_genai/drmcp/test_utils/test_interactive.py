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

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams
from mcp.types import ElicitResult

from datarobot_genai.drmcp import get_dr_mcp_server_url
from datarobot_genai.drmcp import get_headers
from datarobot_genai.drmcp.test_utils.clients.base import LLMResponse
from datarobot_genai.drmcp.test_utils.clients.base import ToolCall
from datarobot_genai.drmcp.test_utils.clients.dr_gateway import DRLLMGatewayMCPClient

# Re-export for backwards compatibility
__all__ = ["DRLLMGatewayMCPClient", "LLMResponse", "ToolCall", "test_mcp_interactive"]


async def test_mcp_interactive() -> None:
    """Test the MCP server interactively with LLM agent."""
    # Check for required environment variables
    datarobot_api_token = os.environ.get("DATAROBOT_API_TOKEN")
    if not datarobot_api_token:
        print("❌ Error: DATAROBOT_API_TOKEN environment variable is required")
        print("Please set it in your .env file or export it")
        return

    # Optional DataRobot settings
    datarobot_endpoint = os.environ.get("DATAROBOT_ENDPOINT")
    model = os.environ.get("MODEL")

    print("🤖 Initializing LLM MCP Client...")

    # Initialize the LLM client with elicitation handler
    config = {
        "datarobot_api_token": datarobot_api_token,
        "save_llm_responses": False,
    }
    if datarobot_endpoint:
        config["datarobot_endpoint"] = datarobot_endpoint
    if model:
        config["model"] = model

    llm_client = DRLLMGatewayMCPClient(str(config))

    # Get MCP server URL
    mcp_server_url = get_dr_mcp_server_url()
    if not mcp_server_url:
        print("❌ Error: MCP server URL is not configured")
        print("Please set DR_MCP_SERVER_URL environment variable or run: task test-interactive")
        return

    print(f"🔗 Connecting to MCP server at: {mcp_server_url}")

    # Elicitation handler: prompt user for required values
    async def elicitation_handler(
        context: RequestContext[ClientSession, Any], params: ElicitRequestParams
    ) -> ElicitResult:
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
