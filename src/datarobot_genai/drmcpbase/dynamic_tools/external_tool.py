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

import asyncio
import logging
from collections.abc import Callable
from collections.abc import Coroutine
from typing import Any
from typing import Literal
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientTimeout
from aiohttp_retry import ExponentialRetry
from aiohttp_retry import RetryClient
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.tools.tool import ToolResult
from pydantic import BaseModel
from pydantic import Field

from datarobot_genai.drmcpbase.dynamic_tools.schema import create_input_schema_pydantic_model
from datarobot_genai.drmcpbase.dynamic_tools.utils import format_response_as_tool_result

logger = logging.getLogger(__name__)


# HTTP request retry configuration
REQUEST_RETRY_SLEEP = 0.1
REQUEST_MAX_RETRY = 5
REQUEST_RETRYABLE_STATUS_CODES = {429, 570, 502, 503, 504}

# HTTP connection timeouts in seconds
REQUEST_CONNECT_TIMEOUT = 30
REQUEST_TOTAL_TIMEOUT = 600

# Headers that should be forwarded from incoming requests.
# Keep this lower-cased for case-insensitive comparisons.
REQUEST_FORWARDED_HEADERS: set[str] = {
    "x-agent-id",
    "x-datarobot-authorization-context",
}


class ExternalToolRegistrationConfig(BaseModel):
    """Configuration for registering an external HTTP API endpoint as an MCP tool.

    This specification defines how to register a generic external HTTP API as a tool
    that can be called by LLM agents through the MCP (Model Context Protocol) server.
    The tool acts as a bridge between the LLM and any external HTTP API, automatically
    handling request construction, retry logic, and response formatting.
    """

    name: str = Field(..., description="Name of the tool.")
    title: str | None = Field(None, description="Title for LLMs and users.")
    description: str | None = Field(None, description="Description for LLMs and users.")
    method: Literal["GET", "POST", "PATCH", "PUT", "DELETE"] = Field(
        ..., description="HTTP method to use."
    )
    base_url: str = Field(..., description="Base URL of the external API.")
    endpoint: str = Field(
        ...,
        description="URL endpoint/route for the external API, may include path params.",
    )
    headers: dict[str, str] | None = Field(
        None, description="Optional static headers to include in requests."
    )
    input_schema: dict[str, Any] = Field(
        ..., description="Pydantic schema defining the tool's input schema."
    )
    tags: set[str] | None = Field(
        None, description="Optional tags for tool categorization and filtering."
    )


def _external_tool_callable_factory(
    spec: ExternalToolRegistrationConfig,
    allow_empty: bool = False,
) -> Callable[[Any], Coroutine[Any, Any, ToolResult]]:
    """Dynamically creates an async callable that makes HTTP requests
    based on the given spec. This callable is the execution logic of the
    tool making the external API call.

    Args:
        spec: Configuration specifying how to make the HTTP request.
        allow_empty: Whether to allow tools with no input parameters.

    Returns
    -------
        An async callable that takes validated inputs and returns a ToolResult.
    """
    input_model = create_input_schema_pydantic_model(
        input_schema=spec.input_schema,
        allow_empty=allow_empty,
    )

    async def call_external_tool(inputs: input_model) -> ToolResult:  # type: ignore[valid-type]
        request_input = inputs.model_dump()  # type: ignore[attr-defined]

        # Extract request parameters
        path_params = request_input.get("path_params", {})
        params = request_input.get("query_params")
        data = request_input.get("data")
        json = request_input.get("json")
        headers = await get_outbound_headers(spec)

        # Build full URL with path params
        url = urljoin(spec.base_url, spec.endpoint.format(**path_params))

        # Configure timeouts
        client_timeout = ClientTimeout(
            total=REQUEST_TOTAL_TIMEOUT,
            connect=REQUEST_CONNECT_TIMEOUT,
        )

        # Configure retry strategy with exponential backoff
        retry_options = ExponentialRetry(
            attempts=REQUEST_MAX_RETRY,
            start_timeout=REQUEST_RETRY_SLEEP,
            statuses=REQUEST_RETRYABLE_STATUS_CODES,
            exceptions={
                aiohttp.ClientError,
                aiohttp.ServerTimeoutError,
                asyncio.TimeoutError,
            },
        )

        # Execute request with retry logic
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            retry_client = RetryClient(
                client_session=session,
                retry_options=retry_options,
                logger=logger,
            )

            async with retry_client.request(
                method=spec.method.upper(),
                url=url,
                params=params,
                data=data,
                json=json,
                headers=headers,
            ) as response:
                content = await response.read()

                if response.status >= 400:
                    error_body = content.decode(response.charset or "utf-8", errors="replace")
                    logger.warning(
                        f"External tool request failed with status {response.status}",
                        extra={
                            "url": url,
                            "error_body": error_body,
                        },
                    )
                    raise ToolError(f"HTTP {response.status} error from deployment: {error_body}")

                return format_response_as_tool_result(
                    data=content,
                    content_type=response.content_type,
                    charset=response.charset or "utf-8",
                )

    return call_external_tool


async def get_outbound_headers(spec: ExternalToolRegistrationConfig) -> dict[str, str]:
    """Retrieve headers to send to the external tool.

    Forwards whitelisted headers from the current FastMCP HTTP request,
    merged with tool-specific static headers. Spec headers always win
    on case-insensitive key collisions.
    """
    headers = get_http_headers()

    # Headers from the incoming request to be forwarded (case-insensitive match)
    forwarded_headers: dict[str, str] = {
        key: value for key, value in headers.items() if key.lower() in REQUEST_FORWARDED_HEADERS
    }

    spec_headers = spec.headers or {}

    # Spec headers take priority; forwarded headers fill in gaps (case-insensitive dedup)
    spec_lower = {k.lower() for k in spec_headers}
    out_headers: dict[str, str] = dict(spec_headers)
    for key, value in forwarded_headers.items():
        if key.lower() not in spec_lower:
            out_headers[key] = value

    return out_headers
