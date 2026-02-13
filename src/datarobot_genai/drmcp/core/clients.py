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

import contextvars
import logging
from typing import Any
from typing import cast

import datarobot as dr
from datarobot.context import Context as DRContext
from datarobot.rest import RESTClientObject
from fastmcp.server.dependencies import get_http_headers
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler
from datarobot_genai.core.utils.auth import DRAppCtx

from .constants import MCP_PATH_ENDPOINT
from .credentials import get_credentials
from .routes_utils import prefix_mount_path

logger = logging.getLogger(__name__)

# Context variable for request headers. Set by RequestHeadersMiddleware for every
# HTTP request so get_sdk_client() can resolve the API token in custom routes and tools.
_request_headers_ctx: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "request_headers", default=None
)


def set_request_headers_for_context(headers: dict[str, str]) -> None:
    """Set request headers in context so get_sdk_client() can use them (e.g. in tests)."""
    _request_headers_ctx.set(headers)


def _resolve_token_from_headers() -> str | None:
    """
    Resolve API token from request headers, trying both sources.

    Order: try framework get_http_headers() first (preferred), then context
    (set by RequestHeadersMiddleware). If both have headers, tries token extraction
    from the first; if no token found, tries the second so the token is used
    whichever source it came from.
    """
    framework_headers = None
    try:
        framework_headers = get_http_headers()
    except Exception:
        pass  # No HTTP context (e.g. stdio transport)

    request_headers_ctx = _request_headers_ctx.get()

    token = extract_token_from_headers(framework_headers) if framework_headers else None
    if not token and request_headers_ctx:
        token = extract_token_from_headers(request_headers_ctx)
    return token


class RequestHeadersMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that sets request headers in context for custom REST routes only.

    Skips the streamable-http MCP path so only routes in routes.py get headers in
    context; get_sdk_client() can then resolve the API token for those handlers.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        mcp_path = prefix_mount_path(MCP_PATH_ENDPOINT).rstrip("/") or "/"
        path = request.url.path
        if path == mcp_path or path.startswith(mcp_path + "/"):
            return await call_next(request)
        headers = {k.lower(): v for k, v in request.headers.items()}
        set_request_headers_for_context(headers)
        return await call_next(request)


# Header names to check for authorization tokens (in order of preference)
HEADER_TOKEN_CANDIDATE_NAMES = [
    "authorization",
    "x-datarobot-api-token",
    "x-datarobot-api-key",
]


def _extract_token_from_headers(headers: dict[str, str]) -> str | None:
    """
    Extract a Bearer token from headers by checking multiple header name candidates.

    Args:
        headers: Dictionary of headers (keys should be lowercase)

    Returns
    -------
        The extracted token string, or None if not found
    """
    for candidate_name in HEADER_TOKEN_CANDIDATE_NAMES:
        auth_header = headers.get(candidate_name)
        if not auth_header:
            continue

        if not isinstance(auth_header, str):
            continue

        # Handle Bearer token format
        bearer_prefix = "bearer "
        if auth_header.lower().startswith(bearer_prefix):
            token = auth_header[len(bearer_prefix) :].strip()
        else:
            # Assume it's a plain token
            token = auth_header.strip()

        if token:
            return token

    return None


def _extract_token_from_auth_context(headers: dict[str, str]) -> str | None:
    """
    Extract API token from authorization context metadata as a fallback.

    Args:
        headers: Dictionary of headers (keys should be lowercase)

    Returns
    -------
        The extracted API key from auth context metadata, or None if not found
    """
    try:
        auth_handler = AuthContextHeaderHandler()

        auth_ctx = auth_handler.get_context(headers)
        if not auth_ctx or not auth_ctx.metadata:
            return None

        metadata = auth_ctx.metadata
        if not isinstance(metadata, dict):
            return None

        dr_ctx: DRAppCtx = DRAppCtx(**metadata.get("dr_ctx", {}))
        if dr_ctx.api_key:
            logger.debug("Extracted token from auth context")
            return dr_ctx.api_key

        return None

    except Exception as e:
        logger.debug(f"Failed to get token from auth context: {e}")
        return None


def extract_token_from_headers(headers: dict[str, str]) -> str | None:
    """
    Extract a token from headers with multiple fallback strategies.

    This function attempts to extract a token in the following order:
    1. From standard authorization headers (Bearer token, x-datarobot-api-token, etc.)
    2. From authorization context metadata (dr_ctx.api_key)

    Args:
        headers: Dictionary of headers (keys should be lowercase)

    Returns
    -------
        The extracted token string, or None if not found
    """
    if token := _extract_token_from_headers(headers):
        return token

    if token := _extract_token_from_auth_context(headers):
        return token

    return None


def get_sdk_client() -> Any:
    """
    Get a DataRobot SDK client using the API token from the current request.

    Token resolution (prefer framework first, then our context):
    1. get_http_headers() (framework; e.g. MCP tool request context)
    2. _request_headers_ctx (set by RequestHeadersMiddleware or tests)
    Token extraction from each source tries: standard auth headers, then
    authorization context metadata (dr_ctx.api_key). If both sources have
    headers, the token is taken from the first source that yields one.

    Raises
    ------
        ValueError: If no API token is found in either header source.

    Note
    ----
        Used in both Global MCP and DR MCP deployments; consider impact before changing.
    """
    token = _resolve_token_from_headers()
    if token:
        logger.debug("Using API token from request headers")
    else:
        logger.warning(
            "No API token found in HTTP headers or authorization context "
            "(e.g. stdio transport or missing Authorization header)."
        )

    if not token:
        raise ValueError("No API token found in HTTP headers or authorization context")

    credentials = get_credentials()
    dr.Client(token=token, endpoint=credentials.datarobot.endpoint)
    # Avoid use-case context from trafaret affecting tool calls
    DRContext.use_case = None
    return dr


def get_api_client() -> RESTClientObject:
    """Get a DataRobot SDK api client using application credentials."""
    dr = get_sdk_client()

    return cast(RESTClientObject, dr.client.get_client())


def get_s3_bucket_info() -> dict[str, str]:
    """Get S3 bucket configuration."""
    credentials = get_credentials()
    return {
        "bucket": credentials.aws_predictions_s3_bucket,
        "prefix": credentials.aws_predictions_s3_prefix,
    }
