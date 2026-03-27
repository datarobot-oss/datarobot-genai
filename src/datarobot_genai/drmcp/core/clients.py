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

import logging
from typing import Any
from typing import cast

import datarobot as dr
from datarobot.context import Context as DRContext
from datarobot.rest import RESTClientObject
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from datarobot_genai.drtools.core.auth import resolve_token_from_headers
from datarobot_genai.drtools.core.auth import set_request_headers_for_context
from datarobot_genai.drtools.core.credentials import get_credentials

from .constants import MCP_PATH_ENDPOINT
from .routes_utils import prefix_mount_path

logger = logging.getLogger(__name__)


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


def get_sdk_client(headers_auth_only: bool = False) -> Any:
    """
    Get a DataRobot SDK client using the API token from the current request.

    Token resolution (prefer framework first, then our context):
    1. get_http_headers() (framework; e.g. MCP tool request context)
    2. _request_headers_ctx (set by RequestHeadersMiddleware or tests)
    Token extraction from each source tries: standard auth headers, then
    authorization context metadata (dr_ctx.api_key). If both sources have
    headers, the token is taken from the first source that yields one.
    3. Application credentials as final fallback

    If headers_auth_only is True, only use the token from the request headers.

    Note
    ----
        Use this function to get a DataRobot SDK client for use in core modules and routes only.
        For use in tools, use DataRobotClient class in tools/clients/datarobot.py.
    """
    token = resolve_token_from_headers()
    if token:
        logger.debug("Using API token from request headers")

    credentials = get_credentials()

    if headers_auth_only and not token:
        raise ValueError("No API token found in HTTP headers")

    # Fallback: Use application token
    if not token:
        # required for dynamic tool and prompt registration on startup
        token = credentials.datarobot.application_api_token
        logger.debug("Using application API token from credentials")

    dr.Client(token=token, endpoint=credentials.datarobot.endpoint)
    # Avoid use-case context from trafaret affecting tool calls
    DRContext.use_case = None
    return dr


def get_api_client(headers_auth_only: bool = False) -> RESTClientObject:
    """Get a DataRobot SDK api client using application credentials."""
    dr = get_sdk_client(headers_auth_only=headers_auth_only)

    return cast(RESTClientObject, dr.client.get_client())


def setup_and_return_dr_api_client_with_static_config_in_container() -> RESTClientObject:
    credentials = get_credentials()

    return dr.Client(
        token=credentials.datarobot.application_api_token, endpoint=credentials.datarobot.endpoint
    )
