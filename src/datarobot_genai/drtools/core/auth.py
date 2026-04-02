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


import contextvars
import logging
from typing import Any

from datarobot.auth.session import AuthCtx
from datarobot.models.genai.agent.auth import ToolAuth

from datarobot_genai.core.utils.auth import AsyncOAuthTokenProvider
from datarobot_genai.core.utils.auth import AuthContextHeaderHandler
from datarobot_genai.core.utils.auth import DRAppCtx
from datarobot_genai.drtools.core.constants import AUTH_CTX_KEY
from datarobot_genai.drtools.core.constants import HEADER_TOKEN_CANDIDATE_NAMES

# Try to import get_http_headers from FastMCP if available
# The deplyment is expected to have the fastmcp dependency installed.
# If not, you need to add your own implementation of get_http_headers.
try:
    from fastmcp.server.dependencies import get_context
    from fastmcp.server.dependencies import get_http_headers
    from fastmcp.server.middleware import Middleware

    _get_http_headers = get_http_headers
    _get_context = get_context
except ImportError:
    # FastMCP not available - create a stub that returns empty dict
    def _get_http_headers(include_all: bool = False) -> dict[str, str]:
        """
        Stub implementation when FastMCP is not available.
        Returns empty dict to match FastMCP behavior when no request context exists.
        """
        return {}

    def _get_context() -> Any:
        """
        Stub implementation when FastMCP is not available.
        Returns empty context to match FastMCP behavior when no request context exists.
        """
        return None


logger = logging.getLogger(__name__)

# Context variable for request headers. Set by RequestHeadersMiddleware for every
# HTTP request so get_sdk_client() can resolve the API token in custom routes and tools.
_request_headers_ctx: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "request_headers", default=None
)


class OAuthMiddleWare(Middleware):
    """Middleware that parses `x-datarobot-authorization-context` for tool calls.

    The header is expected to be a JWT-encoded token representing an authentication
    context compatible with :class:`datarobot.auth.session.AuthCtx`.

    Attributes
    ----------
    auth_handler : AuthContextHeaderHandler
        Handler for encoding/decoding JWT tokens containing auth context.
    """

    def __init__(self, auth_handler: AuthContextHeaderHandler | None = None) -> None:
        self.auth_handler = auth_handler or AuthContextHeaderHandler()

    async def on_call_tool(self, context: Any, call_next: Any) -> Any:
        """Parse header and attach an AuthCtx to the context before running the tool.

        Parameters
        ----------
        context : MiddlewareContext
            The middleware context that will be passed to the tool.
        call_next : CallNext[Any, ToolResult]
            The next handler in the middleware chain.

        Returns
        -------
        ToolResult
            The result from the tool execution.
        """
        auth_context = self._extract_auth_context()
        if not auth_context:
            logger.debug("No valid authorization context extracted from request headers.")

        if context.fastmcp_context is not None:
            await context.fastmcp_context.set_state(AUTH_CTX_KEY, auth_context)
            logger.debug("Authorization context attached to state.")

        return await call_next(context)

    def _extract_auth_context(self) -> AuthCtx | None:
        """Extract and validate authentication context from request headers.

        Returns
        -------
        Optional[AuthCtx]
            The validated authentication context, or None if extraction fails.
        """
        try:
            headers = _get_http_headers()
            return self.auth_handler.get_context(headers)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Failed to extract auth context from headers: %s", exc, exc_info=True)
            return None
        except Exception as exc:
            logger.error("Unexpected error extracting auth context: %s", exc, exc_info=True)
            return None


async def must_get_auth_context() -> AuthCtx:
    """Retrieve the AuthCtx from the current request context or raise error.

    Raises
    ------
    RuntimeError
        If no authorization context is found in the request.

    Returns
    -------
    AuthCtx
        The authorization context associated with the current request.
    """
    context = _get_context()

    auth_ctx = await context.get_state(AUTH_CTX_KEY)
    if not auth_ctx:
        raise RuntimeError("Could not retrieve authorization context from FastMCP context state.")

    return auth_ctx


async def get_access_token(provider_type: str | None = None) -> str:
    """Retrieve access token from the DataRobot OAuth Provider Service.

    OAuth access tokens can be retrieved only for providers where the user completed
    the OAuth flow and granted consent.

    Note:
        *   Currently, only On-Behalf-Of (OBO) tokens are supported, which allow tools to
            act on behalf of the authenticated user, after the user has granted his consent.

    Parameters
    ----------
    provider_type : str, optional
        The name of the OAuth provider. It should match the name of the provider configured
        during provider setup. If no value is provided and only one OAuth provider exists, that
        provider will be used. If multiple providers exist and none is specified, an error will be
        raised.

    Returns
    -------
    The oauth access token.
    """
    auth_ctx = await must_get_auth_context()
    logger.debug("Retrieved authorization context")

    oauth_token_provider = AsyncOAuthTokenProvider(auth_ctx)
    oauth_access_token = await oauth_token_provider.get_token(
        auth_type=ToolAuth.OBO,
        provider_type=provider_type,
    )
    return oauth_access_token


def initialize_oauth_middleware(mcp: Any) -> None:
    """Initialize and register OAuth middleware with the MCP server.

    Parameters
    ----------
    mcp : FastMCP
        The FastMCP server instance to register the middleware with.
    secret_key : Optional[str]
        Secret key for JWT validation. If None, uses the value from config.
    """
    mcp.add_middleware(OAuthMiddleWare())
    logger.info("OAuth middleware registered successfully")


def set_request_headers_for_context(headers: dict[str, str]) -> None:
    """Set request headers in context so get_sdk_client() can use them (e.g. in tests)."""
    _request_headers_ctx.set(headers)


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


def _extract_token_from_headers_with_fallback(headers: dict[str, str]) -> str | None:
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


def resolve_token_from_headers() -> str | None:
    """
    Resolve API token from request headers, trying both sources.

    Order: try framework get_http_headers() first (preferred), then context
    (set by RequestHeadersMiddleware). If both have headers, tries token extraction
    from the first; if no token found, tries the second so the token is used
    whichever source it came from.
    """
    framework_headers = None
    try:
        framework_headers = _get_http_headers()
    except Exception:
        pass  # No HTTP context (e.g. stdio transport)

    request_headers_ctx = _request_headers_ctx.get()

    token = (
        _extract_token_from_headers_with_fallback(framework_headers) if framework_headers else None
    )
    if not token and request_headers_ctx:
        token = _extract_token_from_headers_with_fallback(request_headers_ctx)
    return token


def get_api_key_from_headers(header_name: str) -> str | None:
    headers = _get_http_headers()

    candidates = [header_name]

    if header_name.startswith("x-") and not header_name.startswith("x-datarobot-"):
        candidates.append(f"x-datarobot-{header_name[2:]}")
    elif not header_name.startswith("x-datarobot-"):
        candidates.append(f"x-datarobot-{header_name}")

    for name in candidates:
        if value := headers.get(name):
            return value

    return None
