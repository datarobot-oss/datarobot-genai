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

"""Shared auth helpers for new REST routes (toolGallery/tools, toolGallery/toolSets).

``OAuthMiddleWare`` only runs for MCP tool calls (on_call_tool).
Custom REST routes receive headers via ``RequestHeadersMiddleware`` but must
derive identity themselves — this module provides that helper.

Tech spec §3.0.
"""

from __future__ import annotations

from starlette.responses import JSONResponse

from datarobot_genai.drmcputils.auth import extract_auth_context_from_headers
from datarobot_genai.drmcputils.auth import get_request_headers


def get_created_by() -> str | None:
    """Extract the caller's user ID from the current request context.

    Returns ``None`` when the authorization context is absent or unparseable
    (e.g. local dev without auth headers).
    """
    headers = get_request_headers()
    auth_ctx = extract_auth_context_from_headers(headers)
    if auth_ctx and auth_ctx.user:
        return auth_ctx.user.id
    return None


def require_created_by() -> str | JSONResponse:
    """Return the caller's user ID or a 401 ``JSONResponse``.

    Usage in a REST handler::

        result = require_created_by()
        if isinstance(result, JSONResponse):
            return result
        created_by: str = result
    """
    created_by = get_created_by()
    if not created_by:
        return JSONResponse(
            status_code=401,
            content={"error": "missing or invalid authorization context"},
        )
    return created_by
