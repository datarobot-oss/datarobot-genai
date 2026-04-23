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
from pydantic import Field

class OktaA2AServerAuthConfig(AuthProviderBaseConfig, name="okta_server_auth"):
    """NAT A2A server-side authentication configuration.

    This configuration is surfaced in the agent's AgentCard, specifically in the
    `authSchemes` section. It should expose the information needed to mint the token
    used to communicate with this agent (`token_url`, `audience`, and `scopes`)
    during the second step of the two-step token exchange, where the ID-JAG token is
    exchanged for the downstream scoped token required to call the agent.
    """

    okta_domain: str = Field(
        description="Okta organisation URL (e.g. https://your-org.okta.com).",
    )
    authorization_server_id: str = Field(
        default="default",
        description="Custom authorisation server ID (e.g. 'ausscimnps4vnh9zE1d7').",
    )
    audience: str | None = Field(
        default=None,
        description=(
            "Expected audience (`aud`) claim for this API."
        ),
    )
    scopes: list[str] = Field(
        default=["read_data"],
        description="Scopes required by this API. Validation ensures the token grants all listed scopes.",
    )

    @property
    def token_url(self) -> str:
        return f"{self.okta_domain}/oauth2/{self.authorization_server_id}/v1/token"