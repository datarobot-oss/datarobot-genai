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

"""Wire drmcpbase FastMCP middleware to drtools auth resolution."""

from typing import Any

from datarobot_genai.drmcpbase.middleware import AuthContextExtractor
from datarobot_genai.drmcpbase.middleware import OAuthMiddleWare
from datarobot_genai.drmcpbase.middleware import register_oauth_middleware
from datarobot_genai.drmcputils.auth import extract_auth_context_from_headers
from datarobot_genai.drmcputils.auth import set_auth_context
from datarobot_genai.drmcputils.auth import set_request_headers
from datarobot_genai.drmcputils.constants import AUTH_CTX_KEY


def create_oauth_middleware(
    extract_auth_context: AuthContextExtractor | None = None,
) -> OAuthMiddleWare:
    """Build OAuth middleware wired to drtools request/auth context injection."""
    return OAuthMiddleWare(
        inject_headers=set_request_headers,
        extract_auth_context=extract_auth_context or extract_auth_context_from_headers,
        set_auth_context=set_auth_context,
        auth_context_state_key=AUTH_CTX_KEY,
    )


def initialize_oauth_middleware(mcp: Any) -> None:
    """Register OAuth middleware with the template MCP server."""
    register_oauth_middleware(mcp, create_oauth_middleware())
