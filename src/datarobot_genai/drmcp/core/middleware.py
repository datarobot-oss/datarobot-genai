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

import logging
from enum import Enum
from enum import auto
from http import HTTPStatus
from typing import Any

from fastmcp.server.auth import AccessToken
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from starlette.authentication import AuthCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import Response
from starlette.types import Scope

from datarobot_genai.drmcp.core.config import get_config
from datarobot_genai.drmcpbase.auth.jwt import JWTTokenClaimsValidator
from datarobot_genai.drmcpbase.auth.jwt import JWTTokenHandler
from datarobot_genai.drmcpbase.middleware import AuthContextExtractor
from datarobot_genai.drmcpbase.middleware import OAuthMiddleWare
from datarobot_genai.drmcpbase.middleware import register_oauth_middleware
from datarobot_genai.drmcputils.auth import extract_auth_context_from_headers
from datarobot_genai.drmcputils.auth import set_auth_context
from datarobot_genai.drmcputils.auth import set_request_headers
from datarobot_genai.drmcputils.constants import AUTH_CTX_KEY
from datarobot_genai.drmcputils.exceptions import AudienceClaimValidationError

from .routes_utils import prefix_mount_path

logger = logging.getLogger(__name__)


def _normalize_path(path: str) -> str:
    return path.rstrip("/") or "/"


def is_path_exempt_from_oauth_validation(path: str) -> bool:
    """Please refer datarobot_genai/drmcp/core/routes.py for the route definition."""
    path = _normalize_path(path)
    health_path = _normalize_path(prefix_mount_path("/"))
    well_known_prefix = _normalize_path(prefix_mount_path("/.well-known"))
    return path == health_path or path.startswith(well_known_prefix + "/")


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


class ErrorResponse(Enum):
    INVALID_JWT_TOKEN = auto()
    INVALID_OAUTH_AUDIENCE_CLAIM = auto()

    def to_starlette_response(self, message: str | None = None) -> JSONResponse:
        message = message or self.to_default_message()
        return JSONResponse(status_code=self.to_status_code(), content={"detail": message})

    def to_status_code(self) -> int:
        mapping = {
            self.INVALID_OAUTH_AUDIENCE_CLAIM: HTTPStatus.UNAUTHORIZED,
            self.INVALID_JWT_TOKEN: HTTPStatus.UNPROCESSABLE_ENTITY,
        }
        return mapping[self]

    def to_default_message(self) -> str:
        mapping = {
            self.INVALID_OAUTH_AUDIENCE_CLAIM: "Audience claim validation failed.",
            self.INVALID_JWT_TOKEN: "Invalid JWT token.",
        }
        return mapping[self]


class OAuthJWTTokenHandlerMiddleware(BaseHTTPMiddleware):
    """ASGI middleware parsing OAuth JWT token and setting it in request scope.
    The parsed JWT token can be reused in the downstream validation (e.g., audience, tool scope).
    """

    HTTP_HEADER_TO_VALIDATE = "x-datarobot-external-access-token"

    @classmethod
    def to_run_jwt_token_handling(cls, request: Request) -> bool:
        is_oauth_claim_validation_enabled = get_config().oauth_claim_validation
        return is_oauth_claim_validation_enabled and not is_path_exempt_from_oauth_validation(
            request.url.path
        )

    @staticmethod
    def update_scope_with_authenticated_user(
        scope: Scope,
        access_token: AccessToken,
    ) -> None:
        scope["user"] = AuthenticatedUser(access_token)

    @staticmethod
    def update_scope_with_auth_credentials(
        scope: Scope,
        access_token: AccessToken,
    ) -> None:
        scope["auth"] = AuthCredentials(access_token.scopes)

    async def dispatch(
        self,
        request: Request,
        call_next: Any,
    ) -> Response:
        if not self.to_run_jwt_token_handling(request):
            return await call_next(request)

        access_token = JWTTokenHandler.parse_to_access_token(
            self.HTTP_HEADER_TO_VALIDATE,
            request.headers,
        )
        if not access_token:
            return ErrorResponse.INVALID_JWT_TOKEN.to_starlette_response()

        scope = request.scope
        self.update_scope_with_auth_credentials(scope, access_token)
        self.update_scope_with_authenticated_user(scope, access_token)

        return await call_next(request)


class GeneralOAuthClaimValidationMiddleware(BaseHTTPMiddleware):
    """ASGI middleware validating the audience claim in a JWT token. The naming general in this
    context means claim validations in this middleware is not MCP protocol specific check to be only
    triggerd by a specific MCP protocol (e.g., call a tool).
    This middleware is expected to be triggered after OAuthJWTTokenHandlerMiddleware which
    parses JWT token and sets it the request scope which is to be validated in this middleware.
    """

    @staticmethod
    def get_expected_audience_claim() -> str | None:
        mcp_server_config = get_config()
        return mcp_server_config.mcp_xaa_token_audience

    @staticmethod
    def get_user_from_request_scope(request: Request) -> AuthenticatedUser | None:
        return request.scope.get("user")

    async def dispatch(
        self,
        request: Request,
        call_next: Any,
    ) -> Response:
        user = self.get_user_from_request_scope(request)
        if not user:
            message = (
                "No AuthenticatedUser is found in inbound request scope. "
                f"Skip {__class__.__name__}."
            )
            logger.info(message)
            return await call_next(request)

        try:
            claims_validator = JWTTokenClaimsValidator(user)
            claims_validator.validate_audience_claim(self.get_expected_audience_claim())
        except AudienceClaimValidationError as ex:
            error_message = str(ex)
            logger.info(error_message)
            return ErrorResponse.INVALID_OAUTH_AUDIENCE_CLAIM.to_starlette_response(error_message)

        return await call_next(request)
