# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from nat.data_models.authentication import AuthProviderBaseConfig
from pydantic import BaseModel
from pydantic import Field


class OAuth2PassportRequirement(BaseModel):
    """Step 1: SDK prerequisite — ID-JAG passport JWT trusted issuer."""

    trusted_issuer: str = Field(
        description=(
            "Issuer URL of the internal passport JWT that must be validated before "
            "the downstream Okta token exchange."
        ),
    )


class OAuth2TokenExchangePayload(BaseModel):
    """Step 2: RFC 8693 token exchange POST body parameters for the authorization server."""

    audience: str = Field(description="Expected resource / audience (`aud`) for the issued token.")
    subject_token_type: str = Field(
        description=(
            "`subject_token_type` value for RFC 8693 (e.g. JWT access token vs id token URN)."
        ),
    )
    requested_token_type: str = Field(
        description="`requested_token_type` per RFC 8693 (usually an access token URN).",
    )
    token_endpoint_auth_method: str = Field(
        description="OAuth 2.0 token endpoint client authentication method (e.g. private_key_jwt).",
    )


class OAuth2TokenExchangeConfig(AuthProviderBaseConfig):
    """OAuth2 server-side auth for RFC 8693 token exchange surfaced on the AgentCard.

    OpenAPI-relevant fields (`token_url`, `scopes`) populate ``securitySchemes.oauth2``.
    Step 1 (`passport_requirement`) and Step 2 (`exchange_payload`) are advertised in
    ``capabilities.extensions`` for SDKs executing the two-step flow.
    """

    token_url: str = Field(
        description=(
            "Token URL for Token Exchange (RFC 8693). This is the endpoint to which "
            "the client sends the exchange request for a scoped token for this agent."
        ),
    )
    scopes: list[str] = Field(
        default=["read_data"],
        description=(
            "Scopes advertised for this API. Listed under OpenAPI flows; validation ensures "
            "the token grants required scopes."
        ),
    )
    passport_requirement: OAuth2PassportRequirement = Field(
        description="Step 1: trusted issuer for the prerequisite ID-JAG passport JWT.",
    )
    exchange_payload: OAuth2TokenExchangePayload = Field(
        description="Step 2: Okta/token endpoint parameters for the token exchange POST.",
    )
