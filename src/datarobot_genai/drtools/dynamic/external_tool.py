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

"""Framework-agnostic factory for HTTP-backed external tool callables.

The ``build_external_tool_callable`` factory produces an async callable that
performs an outbound HTTP request matching an :class:`ExternalToolRegistrationConfig`
specification. It is host-agnostic — the response formatter and tool-error
exception class are injected so that callers can plug in their own MCP
framework's primitives (FastMCP, raw mcp, ...).
"""

import asyncio
import logging
from collections.abc import Callable
from collections.abc import Coroutine
from collections.abc import MutableMapping
from typing import Any
from typing import Literal
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientTimeout
from aiohttp_retry import ExponentialRetry
from aiohttp_retry import RetryClient
from pydantic import BaseModel
from pydantic import Field
from requests.structures import CaseInsensitiveDict  # type: ignore[import-untyped]

from datarobot_genai.drtools.core.exceptions import ToolError as DrToolsToolError
from datarobot_genai.drtools.dynamic.schema import create_input_schema_pydantic_model

# fastmcp is an optional dependency in drtools. The check_imports.py script
# whitelists ``fastmcp.server.dependencies`` and ``fastmcp.server.middleware``
# so we use that whitelist to fetch ``get_http_headers``. Fall back to a stub
# when fastmcp is not installed (e.g. when this module is reused by hosts that
# do not run on FastMCP).
try:
    from fastmcp.server.dependencies import get_http_headers as _fastmcp_get_http_headers

    def _get_http_headers() -> dict[str, str]:
        return _fastmcp_get_http_headers()
except ImportError:  # pragma: no cover - exercised by environments without fastmcp

    def _get_http_headers() -> dict[str, str]:
        return {}


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


async def _merge_outbound_headers(
    spec: ExternalToolRegistrationConfig, incoming_headers: dict[str, str]
) -> dict[str, str]:
    """Merge whitelisted incoming HTTP headers with the spec's static headers.

    Pure function — host-agnostic. Re-used by drmcp's wrapper that fetches
    headers from FastMCP's request context.
    """
    forwarded_headers: dict[str, str] = {
        key: value
        for key, value in incoming_headers.items()
        if key.lower() in REQUEST_FORWARDED_HEADERS
    }

    spec_headers = spec.headers or {}

    # Use CaseInsensitiveDict for merging forwarded and spec_headers,
    # to prevent duplicates differing only by case.
    out_headers: MutableMapping[str, str] = CaseInsensitiveDict()

    # Insert spec headers first so their casing is preserved if overridden.
    out_headers.update(spec_headers)
    for key, value in forwarded_headers.items():
        if key not in out_headers:
            out_headers[key] = value

    return dict(out_headers)


async def get_outbound_headers(spec: ExternalToolRegistrationConfig) -> dict[str, str]:
    """Retrieve headers to send to the external tool.

    The method forwards whitelisted headers from the current FastMCP HTTP
    request (when available), with tool-specific headers and user overrides.
    When FastMCP isn't installed, only the spec's static headers are used.
    """
    return await _merge_outbound_headers(spec, _get_http_headers())


def build_external_tool_callable(
    spec: ExternalToolRegistrationConfig,
    *,
    format_response: Callable[[bytes, str, str], Any],
    tool_error_class: type[BaseException] = DrToolsToolError,
    allow_empty_schema: bool = False,
    get_outbound_headers_fn: Callable[
        [ExternalToolRegistrationConfig], Coroutine[Any, Any, dict[str, str]]
    ]
    | None = None,
) -> Callable[[Any], Coroutine[Any, Any, Any]]:
    """Build an async callable that issues the HTTP request described by ``spec``.

    Args:
        spec: HTTP request specification (URL, method, headers, schema, ...).
        format_response: Host-supplied function that turns ``(bytes, content_type,
            charset)`` into whatever the host MCP framework expects from a tool
            (e.g. a ``fastmcp.tools.tool.ToolResult``).
        tool_error_class: Exception class to raise on HTTP errors. Defaults to
            :class:`datarobot_genai.drtools.core.exceptions.ToolError` which is a
            drop-in replacement for ``fastmcp.exceptions.ToolError``. Pass
            FastMCP's ``ToolError`` directly when integrating with FastMCP so the
            framework can render the error to clients.
        allow_empty_schema: Whether to allow registration with an empty input
            schema (no properties). Replaces the previous
            ``MCP_SERVER_TOOL_REGISTRATION_ALLOW_EMPTY_SCHEMA`` global config so
            that this module does not depend on drmcp's runtime config.

    Returns
    -------
        An async callable that takes a validated input model instance and returns
        whatever ``format_response`` returns.
    """
    input_model = create_input_schema_pydantic_model(
        input_schema=spec.input_schema,
        allow_empty=allow_empty_schema,
    )

    headers_fn = get_outbound_headers_fn or get_outbound_headers

    async def call_external_tool(inputs: input_model) -> Any:  # type: ignore[valid-type]
        request_input = inputs.model_dump()  # type: ignore[attr-defined]

        # Extract request parameters
        path_params = request_input.get("path_params", {})
        params = request_input.get("query_params")
        data = request_input.get("data")
        json = request_input.get("json")
        headers = await headers_fn(spec)

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
                    raise tool_error_class(
                        f"HTTP {response.status} error from deployment: {error_body}"
                    )

                return format_response(
                    content,
                    response.content_type,
                    response.charset or "utf-8",
                )

    return call_external_tool
