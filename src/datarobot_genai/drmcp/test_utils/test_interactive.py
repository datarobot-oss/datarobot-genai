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

Point it at ANY MCP server (local, or deployed on DataRobot via directAccess):

    python -m datarobot_genai.drmcp.test_utils.test_interactive \
        --url https://staging.datarobot.com/api/v2/deployments/<id>/directAccess/mcp \
        --model bedrock/anthropic.claude-sonnet-4-6

Falls back to DR_MCP_SERVER_URL / DR_LLM_GATEWAY_MODEL when flags are omitted.
Extra headers can be passed with repeated --header KEY=VALUE flags.
"""

import argparse
import asyncio
import json
import os
import time
import traceback
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams
from mcp.types import ElicitResult

from datarobot_genai.drmcp import DRLLMGatewayMCPClient
from datarobot_genai.drmcp import get_dr_mcp_server_url
from datarobot_genai.drmcp import get_headers
from datarobot_genai.drmcp.test_utils import otel_traces


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive LLM-driven test client for any MCP server."
    )
    parser.add_argument(
        "--url",
        default=None,
        help="MCP server URL (default: DR_MCP_SERVER_URL env var). Works with local "
        "servers and DataRobot deployment directAccess URLs.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM gateway model (default: DR_LLM_GATEWAY_MODEL env var).",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra header to send to the MCP server; repeatable.",
    )
    return parser.parse_args()


def _handle_traces_command(user_input: str, deployment_id: str | None) -> bool:
    """Handle the local ``traces [n]`` / ``trace <id>`` commands.

    Returns True when the input was one of those commands (already handled),
    False when it should go to the LLM instead.
    """
    parts = user_input.split()
    if not parts or parts[0].lower() not in {"traces", "trace"}:
        return False

    if deployment_id is None:
        print(
            "📉 OTEL traces are only available for MCP servers deployed on "
            "DataRobot (directAccess URLs)."
        )
        return True

    try:
        if parts[0].lower() == "trace" and len(parts) > 1:
            detail = otel_traces.fetch_trace(deployment_id, parts[1])
            print(otel_traces.format_trace_tree(detail))
        else:
            limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            traces = otel_traces.fetch_traces(deployment_id, limit=limit)
            print(otel_traces.format_traces_table(traces))
            print("\n(use 'trace <traceId>' to see a trace's span tree)")
    except Exception as exc:  # noqa: BLE001 - diagnostics command, never fatal
        print(f"📉 Could not fetch OTEL traces: {exc}")
    return True


async def test_mcp_interactive(args: argparse.Namespace | None = None) -> None:
    """Test the MCP server interactively with LLM agent."""
    if args is None:
        args = argparse.Namespace(url=None, model=None, header=[])

    # Check for required environment variables
    datarobot_api_token = os.environ.get("DATAROBOT_API_TOKEN")
    if not datarobot_api_token:
        print("❌ Error: DATAROBOT_API_TOKEN environment variable is required")
        print("Please set it in your .env file or export it")
        return

    # Optional DataRobot settings
    datarobot_endpoint = os.environ.get("DATAROBOT_ENDPOINT")
    dr_llm_gateway_model = args.model or os.environ.get("DR_LLM_GATEWAY_MODEL")
    llm_temperature = os.environ.get("LLM_TEMPERATURE")

    print("🤖 Initializing LLM MCP Client...")

    # Initialize the LLM client with elicitation handler
    config = {
        "datarobot_api_token": datarobot_api_token,
        "save_llm_responses": False,
    }
    if datarobot_endpoint:
        config["datarobot_endpoint"] = datarobot_endpoint
    if dr_llm_gateway_model:
        config["model"] = dr_llm_gateway_model
    if llm_temperature is not None:
        config["temperature"] = llm_temperature

    if not config.get("model"):
        print("❌ Error: no LLM model configured")
        print("Set DR_LLM_GATEWAY_MODEL in your .env (same as ETE tests).")
        return

    llm_client = DRLLMGatewayMCPClient(str(config))

    # Get MCP server URL
    mcp_server_url = args.url or get_dr_mcp_server_url()
    if not mcp_server_url:
        print("❌ Error: MCP server URL is not configured")
        print("Please pass --url, set DR_MCP_SERVER_URL, or run: task test-interactive")
        return

    # Base auth headers + directAccess support: DataRobot's deployment proxy
    # consumes the Authorization header for gateway auth, and the MCP app
    # itself reads the user token from x-datarobot-api-token — send both so
    # any server (local or deployed) authenticates. --header flags win last.
    headers = get_headers()
    headers.setdefault("x-datarobot-api-token", datarobot_api_token)
    for item in args.header:
        key, sep, value = item.partition("=")
        if not sep:
            print(f"❌ Error: --header must be KEY=VALUE, got {item!r}")
            return
        headers[key] = value

    deployment_id = otel_traces.deployment_id_from_url(mcp_server_url)

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
            headers=headers,
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
                if deployment_id is not None:
                    print(
                        "Type 'traces [n]' to list the server's recent OTEL traces, "
                        "'trace <traceId>' for one trace's span tree."
                    )
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

                    if _handle_traces_command(user_input, deployment_id):
                        print("\n" + "=" * 60)
                        continue

                    print("🤖 AI is thinking...")

                    turn_started = time.perf_counter()
                    response = await llm_client.process_prompt_with_mcp_support(
                        prompt=user_input,
                        mcp_session=session,
                    )
                    turn_seconds = time.perf_counter() - turn_started

                    print(f"\n🤖 AI Response (round-trip {turn_seconds:.1f}s):")
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
    print("🚀 Starting Interactive MCP Client Test")
    print()

    asyncio.run(test_mcp_interactive(_parse_args()))
