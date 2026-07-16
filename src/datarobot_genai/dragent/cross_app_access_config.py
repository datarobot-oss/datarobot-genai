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

from enum import StrEnum

from nat.data_models.authentication import AuthProviderBaseConfig
from pydantic import BaseModel
from pydantic import Field


class TokenEndpointAuthMethod(StrEnum):
    """Supported ``token_endpoint_auth_method`` values for Cross-Application Access."""

    PRIVATE_KEY_JWT = "private_key_jwt"


class CrossAppTokenExchange(BaseModel):
    """Step 1 parameters: RFC 8693 Token Exchange prerequisite to obtain the ID-JAG."""

    trusted_issuer: str = Field(
        description="Org-level AS issuer URL. Validates the subject token in Step 1.",
    )
    audience: str = Field(
        description=(
            "Step 1 AS base URL (e.g. ``https://your-org.okta.com/oauth2/<as-id>``). "
            "ID-JAG is fetched from here."
        ),
    )


class CrossAppTokenRequest(BaseModel):
    """Step 2 parameters: RFC 7523 JWT Bearer Grant to obtain the final access token."""

    token_url: str = Field(
        description=(
            "Token endpoint of the resource AS. "
            "Published as ``securitySchemes.oauth2.flows.clientCredentials.tokenUrl``."
        ),
    )
    audience: str | None = Field(
        default=None,
        description=(
            "Final resource identifier for the agent. "
            "Published in ``capabilities.extensions[].params.token_request.audience``."
        ),
    )
    scopes: list[str] = Field(
        default=["read_data"],
        description=(
            "Scopes requested in Step 2. "
            "Published as ``securitySchemes.oauth2.flows.clientCredentials.scopes``."
        ),
    )


class CrossApplicationAccessConfig(AuthProviderBaseConfig):
    """Server-side Cross-Application Access config surfaced on the AgentCard.

    Hybrid RFC 8693 / RFC 7523 flow: Step 1 exchanges the incoming access token for
    an ID-JAG via ``token_exchange.audience``; Step 2 uses the ID-JAG for the final token.

    ``token_request.token_url`` / ``token_request.scopes`` →
    ``securitySchemes.oauth2.flows.clientCredentials`` only.
    All other fields → ``capabilities.extensions.params`` only.
    """

    token_endpoint_auth_method: TokenEndpointAuthMethod = Field(
        default=TokenEndpointAuthMethod.PRIVATE_KEY_JWT,
        description="Client auth method. Published in ``capabilities.extensions[].params``.",
    )
    token_exchange: CrossAppTokenExchange = Field(
        description="RFC 8693 Token Exchange parameters (Step 1: ID-JAG prerequisite).",
    )
    token_request: CrossAppTokenRequest = Field(
        description="RFC 7523 JWT Bearer Grant parameters (Step 2: final access token).",
    )
