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


class CrossAppTokenExchange(BaseModel):
    """Step 1 parameters: RFC 8693 Token Exchange prerequisite to obtain the ID-JAG."""

    trusted_issuer: str = Field(
        description=(
            "Issuer URL of the trusted identity provider that must validate the "
            "subject token before the downstream token exchange. Used as ``issuer`` "
            "in the OAuth2 client configuration and as the base for JWT client "
            "assertion audience claims."
        ),
    )
    audience: str = Field(
        description=(
            "Authorization Server destination for Step 1: the Okta custom AS token "
            "endpoint base URL (e.g. ``https://your-org.okta.com/oauth2/<as-id>``). "
            "This is where the ID-JAG prerequisite is fetched."
        ),
    )


class CrossAppTokenRequest(BaseModel):
    """Step 2 parameters: RFC 7523 JWT Bearer Grant to obtain the final access token."""

    grant_type: str = Field(
        description=(
            "OAuth 2.0 grant type for the final token request. Must be "
            "``urn:ietf:params:oauth:grant-type:jwt-bearer`` per RFC 7523."
        ),
    )


class CrossApplicationAccessConfig(AuthProviderBaseConfig):
    """OAuth2 server-side auth for Cross-Application Access surfaced on the AgentCard.

    Implements a hybrid RFC 8693 / RFC 7523 flow:

    * **Step 1** (RFC 8693 Token Exchange): exchange an incoming access token for
      an ID-JAG prerequisite via the authorization server specified in
      ``token_exchange.audience``.
    * **Step 2** (RFC 7523 JWT Bearer Grant): use the ID-JAG to obtain the final
      scoped agent token.

    **OpenAPI fields** (``token_url``, ``scopes``) populate
    ``securitySchemes.oauth2.flows.clientCredentials`` in the Agent Card.
    They are NOT included in ``capabilities.extensions.params``.

    **Extension fields** (``target_audience``, ``token_endpoint_auth_method``,
    ``token_exchange``, ``token_request``) are placed exclusively in the
    ``capabilities.extensions`` block for SDK consumption.
    """

    token_url: str = Field(
        description=(
            "Token endpoint URL for the final RFC 7523 grant. Published under "
            "OpenAPI ``securitySchemes.oauth2.flows.clientCredentials.tokenUrl``."
        ),
    )
    scopes: list[str] = Field(
        default=["read_data"],
        description=(
            "Scopes advertised for this API. Published under OpenAPI "
            "``securitySchemes.oauth2.flows.clientCredentials.scopes``."
        ),
    )
    target_audience: str = Field(
        description=(
            "Final resource identifier for the agent being called. Published as "
            "``capabilities.extensions[].params.target_audience`` in the Agent Card."
        ),
    )
    token_endpoint_auth_method: str = Field(
        description=(
            "OAuth 2.0 token endpoint client authentication method "
            "(e.g. ``private_key_jwt``). Published in the extension params."
        ),
    )
    token_exchange: CrossAppTokenExchange = Field(
        description="RFC 8693 Token Exchange parameters (Step 1: ID-JAG prerequisite).",
    )
    token_request: CrossAppTokenRequest = Field(
        description="RFC 7523 JWT Bearer Grant parameters (Step 2: final access token).",
    )
