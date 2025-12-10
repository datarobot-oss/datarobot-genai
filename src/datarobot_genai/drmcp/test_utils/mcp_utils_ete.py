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
import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import ClientSession as HttpClientSession
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .utils import load_env

load_env()


def get_dr_mcp_server_url() -> str | None:
    """Get DataRobot MCP server URL."""
    return os.environ.get("DR_MCP_SERVER_URL")


def get_dr_mcp_server_http_url() -> str | None:
    """Get DataRobot MCP server http URL."""
    return os.environ.get("DR_MCP_SERVER_HTTP_URL")


def get_openai_llm_client_config() -> dict[str, str]:
    """Get OpenAI LLM client configuration."""
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_api_base = os.environ.get("OPENAI_API_BASE")
    openai_api_deployment_id = os.environ.get("OPENAI_API_DEPLOYMENT_ID")
    openai_api_version = os.environ.get("OPENAI_API_VERSION")
    save_llm_responses = os.environ.get("SAVE_LLM_RESPONSES", "false").lower() == "true"

    # Check for OpenAI configuration
    if not openai_api_key:
        raise ValueError("Missing required environment variable: OPENAI_API_KEY")
    if (
        openai_api_base and not openai_api_deployment_id
    ):  # For Azure OpenAI, we need additional variables
        raise ValueError("Missing required environment variable: OPENAI_API_DEPLOYMENT_ID")

    config: dict[str, str] = {
        "openai_api_key": openai_api_key,
    }

    if openai_api_base:
        config["openai_api_base"] = openai_api_base
    if openai_api_deployment_id:
        config["openai_api_deployment_id"] = openai_api_deployment_id
    if openai_api_version:
        config["openai_api_version"] = openai_api_version
    config["save_llm_responses"] = str(save_llm_responses)

    return config


def get_headers() -> dict[str, str]:
    # When the MCP server is deployed in DataRobot, we have to include the API token in headers for
    # authentication.
    api_token = os.getenv("DATAROBOT_API_TOKEN")
    headers = {"Authorization": f"Bearer {api_token}"}
    return headers


@asynccontextmanager
async def ete_test_mcp_session(
    additional_headers: dict[str, str] | None = None,
) -> AsyncGenerator[ClientSession, None]:
    """Create an MCP session for each test.

    Parameters
    ----------
    additional_headers : dict[str, str], optional
        Additional headers to include in the MCP session (e.g., auth headers for testing).
    """
    try:
        headers = get_headers()
        if additional_headers:
            headers.update(additional_headers)

        async with streamablehttp_client(url=get_dr_mcp_server_url(), headers=headers) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=5)
                yield session
    except asyncio.TimeoutError:
        raise TimeoutError(f"Check if the MCP server is running at {get_dr_mcp_server_url()}")


@asynccontextmanager
async def ete_test_http_session(
    additional_headers: dict[str, str] | None = None,
) -> AsyncGenerator[HttpClientSession, None]:
    """Create an HTTP session for each test that can connect to MCP custom http routes.

    Parameters
    ----------
    additional_headers : dict[str, str], optional
        Additional headers to include in the HTTP session (e.g., auth headers for testing).
    """
    headers = get_headers()
    if additional_headers:
        headers.update(additional_headers)

    async with ete_test_mcp_session(additional_headers=additional_headers):
        async with aiohttp.ClientSession(
            base_url=get_dr_mcp_server_http_url(), headers=headers
        ) as client:
            yield client
