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
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jwt
from fastmcp.server.auth import AccessToken
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser

from datarobot_genai.drmcpbase.auth.exceptions import AudienceClaimValidationError
from datarobot_genai.drmcpbase.auth.exceptions import MCPToolScopeClaimValidationError

logger = logging.getLogger(__name__)


class JWTTokenHandler:
    @staticmethod
    def get_bearer_token_header(
        request_header_name_case_insensitive: str,
        request_headers_with_lower_case_key: Mapping[str, str],
    ) -> str | None:
        return request_headers_with_lower_case_key.get(request_header_name_case_insensitive.lower())

    @staticmethod
    def get_bearer_token_value(
        bearer_token_header: str,
        bearer_token_header_can_be_missing: bool = True,
    ) -> str | None:
        schema, _, value = bearer_token_header.partition(" ")
        if schema.lower() == "bearer":
            return value
        elif bearer_token_header_can_be_missing:
            return bearer_token_header
        else:
            return None

    @staticmethod
    def is_jwt_token(token_value: str) -> bool:
        try:
            jwt.get_unverified_header(token_value)
            return True
        except jwt.exceptions.DecodeError:
            return False

    @staticmethod
    def get_jwt_payload_without_signature_verification(jwt_token_value: str) -> dict[str, Any]:
        return jwt.decode(jwt_token_value, options={"verify_signature": False})

    @staticmethod
    def extract_scopes(claims: dict[str, Any]) -> list[str]:
        for claim in ["scope", "scp"]:
            if claim in claims:
                if isinstance(claims[claim], str):
                    return claims[claim].split()
                elif isinstance(claims[claim], list):
                    return claims[claim]
        return []

    @staticmethod
    def extract_exp(claims: dict[str, Any]) -> int | None:
        # exp in JWT is a NumericDate type (Seconds Since the Epoch, integer or non-integer).
        exp = claims.get("exp")
        return int(float(exp)) if exp is not None else None

    @staticmethod
    def extract_client_id(claims: dict[str, Any]) -> str:
        return claims.get("client_id") or claims.get("azp") or claims.get("sub") or "unknown"

    @classmethod
    def parse_to_access_token(
        cls,
        request_header_name: str,
        request_headers: Mapping[str, str],
    ) -> AccessToken | None:
        bearer_token_header = cls.get_bearer_token_header(request_header_name, request_headers)
        bearer_token_value = (
            cls.get_bearer_token_value(bearer_token_header) if bearer_token_header else None
        )
        if bearer_token_value and cls.is_jwt_token(bearer_token_value):
            try:
                payload_as_dict = cls.get_jwt_payload_without_signature_verification(
                    bearer_token_value
                )
                return AccessToken(
                    token=bearer_token_value,
                    client_id=cls.extract_client_id(payload_as_dict),
                    scopes=cls.extract_scopes(payload_as_dict),
                    expires_at=cls.extract_exp(payload_as_dict),
                    claims=payload_as_dict,
                )
            except (jwt.exceptions.PyJWTError, ValueError, TypeError):
                logger.error("Failed to decode JWT", exc_info=True)
                return None
        return None


@dataclass
class AuthorizationClaims:
    scopes: frozenset[str]
    # RFC-7519: aud can be a single string or a list (case-sensitive); None when absent.
    audience: str | list[str] | None = None

    @classmethod
    def from_access_token(cls, access_token: AccessToken) -> "AuthorizationClaims":
        return cls(
            audience=access_token.claims.get("aud", None),
            scopes=frozenset(access_token.scopes) if access_token.scopes else frozenset(),
        )

    def audience_is_a_list(self) -> bool:
        return isinstance(self.audience, list)

    def audience_is_none(self) -> bool:
        return self.audience is None

    def contain_expected_audience(self, expected_audience: str) -> bool:
        if isinstance(self.audience, list):
            return expected_audience in self.audience
        return expected_audience == self.audience

    def has_scope_claims(self) -> bool:
        return len(self.scopes) > 0

    def are_expected_scopes_subset_of_scope_claims(self, expected_scopes: frozenset[str]) -> bool:
        return expected_scopes.issubset(self.scopes)


class JWTTokenClaimsValidator:
    def __init__(self, authenticated_user: AuthenticatedUser) -> None:
        self.claims = AuthorizationClaims.from_access_token(authenticated_user.access_token)

    def validate_audience_claim(self, expected_audience_claim: str | None) -> None:
        if expected_audience_claim is None:
            logger.info(
                "Authorization audience claim validation is not executed. "
                "There is no expected audience claim provided."
            )
            return
        if self.claims.audience_is_none():
            error_message = (
                "Authorization audience claim is not found in the inbound JWT bearer token."
            )
            logger.info(error_message)
            raise AudienceClaimValidationError(error_message)

        if not self.claims.contain_expected_audience(expected_audience_claim):  # type: ignore[union-attr]
            raise AudienceClaimValidationError("Authorization audience claim validation failed")

        logger.info("Audience claim validation succeeded.")

    def validate_mcp_tool_scope_claims(self, expected_scopes: frozenset[str] | None) -> None:
        if expected_scopes is None or not len(expected_scopes):
            logger.info(
                "Authorization MCP tool scope claim validation is not executed. "
                "There is no expected scope claim provided."
            )
            return
        if not self.claims.has_scope_claims():
            error_message = (
                "Authorization MCP tool scope claim is not found in the inbound JWT bearer token."
            )
            logger.info(error_message)
            raise MCPToolScopeClaimValidationError(error_message)

        if not self.claims.are_expected_scopes_subset_of_scope_claims(expected_scopes):  # type: ignore[union-attr]
            raise MCPToolScopeClaimValidationError("Authorization scope claim validation failed")

        logger.info("MCP tool scope claim validation succeeded.")
