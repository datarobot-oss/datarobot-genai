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

import datarobot as dr
from datarobot.rest import RESTClientObject
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from datarobot_genai.drtools.core.auth import set_request_headers
from datarobot_genai.drtools.core.credentials import get_credentials

from .constants import MCP_PATH_ENDPOINT
from .routes_utils import prefix_mount_path

logger = logging.getLogger(__name__)


class RequestHeadersMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that sets request headers in context for custom REST routes only.

    Skips the streamable-http MCP path so only routes in routes.py get headers in
    context; drtools token resolution can then use them in those handlers.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        mcp_path = prefix_mount_path(MCP_PATH_ENDPOINT).rstrip("/") or "/"
        path = request.url.path
        if path == mcp_path or path.startswith(mcp_path + "/"):
            return await call_next(request)
        headers = {k.lower(): v for k, v in request.headers.items()}
        set_request_headers(headers)
        return await call_next(request)


def setup_and_return_dr_api_client_with_static_config_in_container() -> RESTClientObject:
    credentials = get_credentials()

    return dr.Client(
        token=credentials.datarobot.datarobot_api_token,
        endpoint=credentials.datarobot.datarobot_endpoint,
    )
