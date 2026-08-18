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

import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import Response

from datarobot_genai.drmcp.core.config import get_config
from datarobot_genai.drmcpbase.middleware import AuthContextExtractor
from datarobot_genai.drmcpbase.middleware import OAuthMiddleWare
from datarobot_genai.drmcpbase.middleware import register_oauth_middleware
from datarobot_genai.drmcputils.auth import JWTTokenClaimsValidator
from datarobot_genai.drmcputils.auth import extract_auth_context_from_headers
from datarobot_genai.drmcputils.auth import set_auth_context
from datarobot_genai.drmcputils.auth import set_request_headers
from datarobot_genai.drmcputils.constants import AUTH_CTX_KEY
from datarobot_genai.drmcputils.exceptions import AudienceClaimValidationError

from .routes_utils import prefix_mount_path

logger = logging.getLogger(__name__)


def _normalize_path(path: str) -> str:
    return path.rstrip("/") or "/"


def is_exempt_from_validation(path: str) -> bool:
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


class GeneralOAuthClaimValidationMiddleware(BaseHTTPMiddleware):
    """ASGI middleware validating the audience claim in a JWT token. It is a general claim
    validator that is triggered on each inbound FastMCP server request.
    """

    HTTP_HEADER_TO_VALIDATE = "x-datarobot-external-access-token"

    @staticmethod
    def get_expected_audience_claim() -> str | None:
        mcp_server_config = get_config()
        return mcp_server_config.mcp_xaa_token_audience

    @classmethod
    def has_header_to_validate(cls, request: Request) -> bool:
        return cls.HTTP_HEADER_TO_VALIDATE in request.headers

    @classmethod
    def to_execute_validation(cls, request: Request) -> bool:
        return not is_exempt_from_validation(request.url.path) and cls.has_header_to_validate(
            request
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Any,
    ) -> Response:
        if not self.to_execute_validation(request):
            return await call_next(request)

        try:
            claims_validator = JWTTokenClaimsValidator(
                self.HTTP_HEADER_TO_VALIDATE,
                request.headers,
            )
            claims_validator.validate_audience_claim(self.get_expected_audience_claim())
        except AudienceClaimValidationError as ex:
            error_message = str(ex)
            logger.info(error_message)
            return ErrorResponse.INVALID_OAUTH_AUDIENCE_CLAIM.to_starlette_response(error_message)
        except jwt.exceptions.PyJWTError as ex:
            error_message = f"Malformed authorization token: {ex}"
            logger.info(error_message)
            return ErrorResponse.INVALID_JWT_TOKEN.to_starlette_response(error_message)

        return await call_next(request)
