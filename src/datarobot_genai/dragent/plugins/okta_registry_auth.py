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

"""Okta for the agent card registry and its L2 cache; DataRobot API key for A2A RPC.

When the inbound request carries an IdP token in ``x-datarobot-external-access-token``,
an ``authenticated_a2a_client`` using this provider looks the agent card up in the
central registry, and reads and writes the registry's L2 MemorySpace cache, with that
token only -- never ``DATAROBOT_API_TOKEN``, and with no fallback to it. Without the
header the registry lookup uses the DataRobot API key as usual.

A2A RPC to the remote agent always uses the DataRobot API key, so ``bearerAuth``
remotes (no cross-application access on their card) keep working.

Example YAML::

    function_groups:
      remote_agent:
        _type: authenticated_a2a_client
        registry:
          workload_id: "64a1b2c3d4e5f6a7b8c9d0e2"
        auth_provider: okta_registry_auth
    authentication:
      okta_registry_auth:
        _type: okta_registry_dr_call

For callers to hit the L2 cache with their own token, the space must be shared with
their organization; see
:class:`~datarobot_genai.dragent.memory_space_cache.RegistryCacheShareSettings`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.cli.register_workflow import register_auth_provider
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred
from pydantic import SecretStr

from datarobot_genai.core.config import resolve_config
from datarobot_genai.dragent.inbound_token import OAUTH_ACCESS_TOKEN_HEADER
from datarobot_genai.dragent.inbound_token import find_idp_token
from datarobot_genai.dragent.plugins.auth_a2a_client import A2ADiscoveryAuthMixin

logger = logging.getLogger(__name__)


def okta_token_from_request() -> str | None:
    """Return the inbound IdP access token, or ``None`` when absent."""
    try:
        headers = dict(Context.get().metadata.headers or {})
    except Exception:
        return None
    return find_idp_token(headers)


class OktaRegistryDRCallAuthProviderConfig(AuthProviderBaseConfig, name="okta_registry_dr_call"):  # type: ignore[call-arg]
    """Registry and L2 cache use the inbound Okta token; A2A RPC uses the DR API key."""


class OktaRegistryAPIKeyAuthProvider(
    A2ADiscoveryAuthMixin,
    AuthProviderBase[OktaRegistryDRCallAuthProviderConfig],
):
    """Forward inbound Okta for the registry and L2 cache; use the DR API key for RPC.

    The class name must contain ``APIKey``: NAT's A2A credential service matches
    providers to an agent card's ``bearerAuth`` (``HTTPAuthSecurityScheme``) scheme by
    class name.
    """

    def registry_caller_token(self) -> str | None:
        """Return the token the registry and L2 cache must use exclusively, if any."""
        return okta_token_from_request()

    async def authenticate_for_discovery(self, user_id: str | None = None) -> dict[str, str]:
        token = okta_token_from_request()
        if not token:
            raise RuntimeError(
                f"No IdP access token in the request (looked in '{OAUTH_ACCESS_TOKEN_HEADER}')."
            )
        logger.info("Forwarding token from '%s' for discovery", OAUTH_ACCESS_TOKEN_HEADER)
        return {"Authorization": f"Bearer {token}"}

    async def authenticate(self, user_id: str | None = None, **kwargs: Any) -> AuthResult | None:
        api_token = resolve_config().resolve_datarobot_api_token()
        if not api_token:
            raise RuntimeError(
                "DATAROBOT_API_TOKEN is required for A2A RPC with okta_registry_dr_call."
            )
        return AuthResult(credentials=[BearerTokenCred(token=SecretStr(api_token))])


@register_auth_provider(config_type=OktaRegistryDRCallAuthProviderConfig)  # type: ignore[untyped-decorator]
async def okta_registry_dr_call_auth(
    config: OktaRegistryDRCallAuthProviderConfig, _builder: Builder
) -> AsyncGenerator[OktaRegistryAPIKeyAuthProvider]:
    yield OktaRegistryAPIKeyAuthProvider(config)
