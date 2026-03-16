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
from typing import Any

import httpx
from a2a.client import Client
from a2a.client import ClientConfig
from a2a.client import ClientFactory
from nat.authentication.interfaces import AuthProviderBase
from nat.builder.context import Context
from nat.data_models.authentication import BearerTokenCred
from nat.data_models.authentication import HeaderCred
from nat.plugins.a2a.client.client_base import A2ABaseClient
from nat.plugins.a2a.client.client_config import A2AClientConfig
from nat.plugins.a2a.client.client_impl import A2AClientFunctionGroup

logger = logging.getLogger(__name__)


def _extract_auth_headers(auth_result: Any) -> dict[str, str]:
    """Extract HTTP headers from an AuthResult, tolerating plain-str tokens."""
    headers: dict[str, str] = {}
    for cred in auth_result.credentials:
        if isinstance(cred, BearerTokenCred):
            headers[cred.header_name] = f"{cred.scheme} {cred.token.get_secret_value()}"
        elif isinstance(cred, HeaderCred):
            headers[cred.name] = cred.value.get_secret_value()

    return headers


class _AuthCardA2ABaseClient(A2ABaseClient):
    """A2ABaseClient that authenticates all requests to the remote agent.

    Overrides ``__aenter__`` to pre-authenticate via the auth provider and embed
    the Bearer token in the ``httpx.AsyncClient`` default headers.  This ensures
    every request — agent-card fetch *and* every ``send_message`` — carries auth,
    without relying on ``AuthInterceptor`` which only fires when the agent card
    defines ``security``/``security_schemes`` (DataRobot deployments don't).
    """

    _httpx_client: httpx.AsyncClient | None
    _client: Client | None

    async def __aenter__(self) -> "_AuthCardA2ABaseClient":
        if self._httpx_client is not None or self._client is not None:
            raise RuntimeError("A2ABaseClient already initialized")

        # Pre-authenticate to embed Bearer token in every httpx request.
        default_headers: dict[str, str] = {}
        if self._auth_provider:
            try:
                user_id = Context.get().user_id or "default-user"
                auth_result = await self._auth_provider.authenticate(user_id=user_id)
                if auth_result:
                    default_headers = _extract_auth_headers(auth_result)
                    logger.info(
                        "Pre-authenticated: injecting %d auth header(s) into httpx client",
                        len(default_headers),
                    )
            except Exception:
                logger.warning("Failed to pre-authenticate for A2A client", exc_info=True)

        self._httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._task_timeout.total_seconds()),
            headers=default_headers,
        )

        # Resolve agent card — auth headers are already baked into the httpx client.
        await self._resolve_agent_card()
        if not self._agent_card:
            raise RuntimeError("Agent card not resolved")

        # Create A2A client without AuthInterceptor — auth is at the httpx level.
        client_config = ClientConfig(
            httpx_client=self._httpx_client,
            streaming=self._streaming,
        )
        factory = ClientFactory(client_config)
        self._client = factory.create(self._agent_card)

        logger.info(
            "Connected to A2A agent at %s with pre-injected auth headers",
            self._base_url,
        )
        return self


class AuthenticatedA2AClientConfig(A2AClientConfig, name="authenticated_a2a_client"):  # type: ignore[call-arg]
    """A2AClientConfig variant that authenticates the agent-card fetch.

    Identical to ``A2AClientConfig`` — the separate ``name`` avoids a
    registration conflict with the upstream ``a2a_client`` function group.

    Use ``_type: authenticated_a2a_client`` in workflow.yaml instead of
    ``_type: a2a_client`` when the remote agent requires authentication
    for the agent-card endpoint.
    """


class _AuthCardA2AClientFunctionGroup(A2AClientFunctionGroup):
    """A2AClientFunctionGroup that uses ``_AuthCardA2ABaseClient`` so the
    agent-card fetch is authenticated.

    The only difference from the base ``__aenter__`` is constructing
    ``_AuthCardA2ABaseClient`` instead of ``A2ABaseClient``.  The base class
    hardcodes that constructor call, so we must override the method.
    """

    async def __aenter__(self) -> "_AuthCardA2AClientFunctionGroup":
        config: A2AClientConfig = self._config  # type: ignore[assignment]
        base_url = str(config.url)

        user_id = Context.get().user_id
        if not user_id:
            raise RuntimeError("User ID not found in context")

        auth_provider: AuthProviderBase | None = None
        if config.auth_provider:
            try:
                auth_provider = await self._builder.get_auth_provider(config.auth_provider)
                logger.info("Resolved authentication provider for A2A client")
            except Exception as e:
                logger.error("Failed to resolve auth provider '%s': %s", config.auth_provider, e)
                raise RuntimeError(f"Failed to resolve auth provider: {e}") from e

        # Only difference from upstream: _AuthCardA2ABaseClient instead of A2ABaseClient
        self._client = _AuthCardA2ABaseClient(
            base_url=base_url,
            agent_card_path=config.agent_card_path,
            task_timeout=config.task_timeout,
            streaming=config.streaming,
            auth_provider=auth_provider,
        )
        await self._client.__aenter__()

        if auth_provider:
            logger.info(
                "Connected to A2A agent at %s with authentication (user_id: %s)", base_url, user_id
            )
        else:
            logger.info("Connected to A2A agent at %s (user_id: %s)", base_url, user_id)

        self._register_functions()
        return self
