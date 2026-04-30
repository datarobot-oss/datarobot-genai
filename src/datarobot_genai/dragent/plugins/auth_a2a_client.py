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

import abc
import logging
from collections.abc import AsyncGenerator
from typing import Any
from typing import Protocol
from typing import runtime_checkable

import httpx
from a2a.client import A2ACardResolver
from a2a.client import AuthInterceptor
from a2a.client import ClientConfig
from a2a.client import ClientFactory
from a2a.types import AgentCard
from nat.authentication.interfaces import AuthProviderBase
from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.cli.register_workflow import register_per_user_function_group
from nat.data_models.authentication import BearerTokenCred
from nat.data_models.authentication import HeaderCred
from nat.plugins.a2a.auth.credential_service import A2ACredentialService
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


@runtime_checkable
class AgentCardAware(Protocol):
    """Auth providers that need the resolved agent card before ``authenticate()``.

    Implement this protocol on an auth provider to receive the fetched
    :class:`~a2a.types.AgentCard` before the first ``authenticate()`` call.
    :class:`_AuthenticatedA2ABaseClient` checks for this protocol via
    ``isinstance`` and calls :meth:`set_agent_card` automatically.
    """

    def set_agent_card(self, card: AgentCard) -> None:
        """Receive the resolved agent card.

        Called by :class:`_AuthenticatedA2ABaseClient` immediately after the
        card is fetched, before ``authenticate()`` is ever invoked.
        """
        ...


class A2ADiscoveryAuthMixin(abc.ABC):
    """Mixin for auth providers that need different credentials for agent card discovery.

    By default :meth:`_AuthenticatedA2ABaseClient._resolve_agent_card` calls
    ``authenticate()`` and extracts the resulting credentials as HTTP headers.
    Implement this mixin to return different headers for the agent card GET
    without affecting the call-phase ``authenticate()`` used by ``AuthInterceptor``.

    Example: :class:`~datarobot_genai.dragent.plugins.okta_a2a_auth.OktaTokenExchangeAuthProvider`
    uses this to forward the incoming Okta bearer token for discovery while
    performing a two-step RFC 8693 exchange for the actual A2A calls.
    """

    @abc.abstractmethod
    async def authenticate_for_discovery(self, user_id: str | None = None) -> dict[str, str]:
        """Return HTTP headers for the agent card GET request.

        Returns
        -------
        dict[str, str]
            Header name → value pairs to attach to the agent card HTTP request.
        """


class _AuthenticatedA2ABaseClient(A2ABaseClient):
    """A2A client that authenticates agent card discovery and A2A RPC independently.

    * **Agent card** ``GET`` — two mutually exclusive paths:

      1. *Discovery mixin*: if the configured ``auth_provider`` implements
         :class:`A2ADiscoveryAuthMixin`, ``authenticate_for_discovery()`` is called
         and its result is used as headers on a short-lived ``httpx.AsyncClient``.
      2. *Default*: ``authenticate()`` is called and its credentials are extracted
         as headers (e.g. ``DataRobotAPIKeyAuthProvider``).  When no provider is
         set the card is fetched unauthenticated.

    * **Task / message** traffic uses NAT's ``AuthInterceptor`` +
      ``A2ACredentialService``, which calls ``authenticate()`` per-request.
    """

    async def _resolve_agent_card(self) -> None:
        user_id = Context.get().user_id or "default-user"

        if isinstance(self._auth_provider, A2ADiscoveryAuthMixin):
            headers = await self._auth_provider.authenticate_for_discovery(user_id)
            logger.info(
                "Fetching agent card (custom discovery auth) from: %s%s",
                self._base_url,
                self._agent_card_path,
            )
        elif self._auth_provider:
            auth_result = await self._auth_provider.authenticate(user_id=user_id)
            headers = _extract_auth_headers(auth_result) if auth_result else {}
            logger.info(
                "Fetching agent card (auth_provider authenticate()) from: %s%s",
                self._base_url,
                self._agent_card_path,
            )
        else:
            headers = {}
            logger.info(
                "Fetching agent card (unauthenticated) from: %s%s",
                self._base_url,
                self._agent_card_path,
            )

        timeout = httpx.Timeout(self._task_timeout.total_seconds())
        try:
            async with httpx.AsyncClient(timeout=timeout, headers=headers) as card_client:
                resolver = A2ACardResolver(
                    httpx_client=card_client,
                    base_url=self._base_url,
                    agent_card_path=self._agent_card_path,
                )
                self._agent_card = await resolver.get_agent_card()
                logger.info("Successfully fetched agent card")
        except Exception as e:
            logger.error("Failed to fetch agent card: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to fetch agent card from {self._base_url}") from e

    async def __aenter__(self) -> "_AuthenticatedA2ABaseClient":
        if self._httpx_client is not None or self._client is not None:  # type: ignore[has-type]
            raise RuntimeError("A2ABaseClient already initialized")

        self._httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._task_timeout.total_seconds()),
        )

        await self._resolve_agent_card()
        if not self._agent_card:
            raise RuntimeError("Agent card not resolved")

        # Allow auth providers that need agent-card parameters (e.g. OktaTokenExchangeAuthProvider)
        # to receive the resolved card before the interceptor is set up.
        if isinstance(self._auth_provider, AgentCardAware):
            self._auth_provider.set_agent_card(self._agent_card)

        interceptors: list[Any] = []
        if self._auth_provider:
            credential_service = A2ACredentialService(
                auth_provider=self._auth_provider,
                agent_card=self._agent_card,
            )
            interceptors.append(AuthInterceptor(credential_service))
            logger.info("Task-phase authentication configured for A2A client (AuthInterceptor)")

        client_config = ClientConfig(
            httpx_client=self._httpx_client,
            streaming=self._streaming,
        )
        factory = ClientFactory(client_config)
        self._client = factory.create(self._agent_card, interceptors=interceptors)

        logger.info("Connected to A2A agent at %s", self._base_url)
        return self


class AuthenticatedA2AClientConfig(A2AClientConfig, name="authenticated_a2a_client"):  # type: ignore[call-arg]
    """A2A client config with separate discovery and call-phase auth.

    Inherits all fields from :class:`~nat.plugins.a2a.client.client_config.A2AClientConfig`
    (``url``, ``auth_provider``, ``agent_card_path``, ``task_timeout``, ``streaming``).

    If the referenced ``auth_provider`` implements :class:`A2ADiscoveryAuthMixin`,
    it supplies different credentials for agent card discovery vs A2A RPC calls.
    Otherwise the same ``authenticate()`` result is used for both phases.
    """


class AuthenticatedA2AClientFunctionGroup(A2AClientFunctionGroup):
    """Uses :class:`_AuthenticatedA2ABaseClient` so both A2A phases are authenticated."""

    async def __aenter__(self) -> "AuthenticatedA2AClientFunctionGroup":
        config: AuthenticatedA2AClientConfig = self._config  # type: ignore[assignment]

        user_id = Context.get().user_id
        if not user_id:
            raise RuntimeError("User ID not found in context")

        auth_provider: AuthProviderBase | None = None
        if config.auth_provider:
            try:
                auth_provider = await self._builder.get_auth_provider(config.auth_provider)
                logger.info(
                    "Resolved authentication provider '%s' for A2A client",
                    config.auth_provider,
                )
            except Exception as e:
                logger.error(
                    "Failed to resolve auth provider '%s': %s",
                    config.auth_provider,
                    e,
                )
                raise RuntimeError(f"Failed to resolve auth provider: {e}") from e

        base_url = str(config.url)
        self._client = _AuthenticatedA2ABaseClient(
            base_url=base_url,
            agent_card_path=config.agent_card_path,
            task_timeout=config.task_timeout,
            streaming=config.streaming,
            auth_provider=auth_provider,
        )
        await self._client.__aenter__()

        logger.info(
            "Connected to A2A agent at %s (auth_provider: %s, user_id: %s)",
            base_url,
            config.auth_provider or "none",
            user_id,
        )

        self._register_functions()
        return self


@register_per_user_function_group(config_type=AuthenticatedA2AClientConfig)  # type: ignore[untyped-decorator]
async def authenticated_a2a_client(
    config: AuthenticatedA2AClientConfig, _builder: Builder
) -> AsyncGenerator[Any, None]:
    """A2A function group with authenticated agent card discovery and RPC.

    The ``auth_provider`` field controls both phases.  If the provider
    implements :class:`A2ADiscoveryAuthMixin` it supplies separate credentials
    for agent card discovery; otherwise the same ``authenticate()`` is used
    for both phases.
    """
    async with AuthenticatedA2AClientFunctionGroup(config, _builder) as group:
        yield group
