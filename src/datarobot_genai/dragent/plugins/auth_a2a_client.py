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
import asyncio
import functools
import inspect
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
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from datarobot_genai.dragent.agent_card_registry import get_default_registry
from datarobot_genai.dragent.agent_card_registry import get_default_registry_sync

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
    """Protocol for auth providers that need the agent card before ``authenticate()``.

    :class:`_AuthenticatedA2ABaseClient` checks for this via ``isinstance``
    and calls :meth:`set_agent_card` automatically after card resolution.
    """

    def set_agent_card(self, card: AgentCard) -> None:
        """Receive the resolved agent card before ``authenticate()`` is invoked."""
        ...


class A2ADiscoveryAuthMixin(abc.ABC):
    """Mixin for auth providers that need different credentials for agent card discovery.

    Without this mixin, ``_AuthenticatedA2ABaseClient`` calls ``authenticate()`` for
    both discovery and call phases.  Implement ``authenticate_for_discovery()`` to
    supply separate headers for the agent card GET.
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
    """A2A client with independent auth for card discovery and RPC calls.

    Discovery uses ``authenticate_for_discovery()`` when the provider implements
    :class:`A2ADiscoveryAuthMixin`, otherwise falls back to ``authenticate()``.
    Task traffic always uses ``AuthInterceptor`` + ``A2ACredentialService``.
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
            logger.error("Failed to fetch agent card from %s: %s", self._base_url, e)
            raise RuntimeError(f"Failed to fetch agent card from {self._base_url}") from e

    async def __aenter__(self) -> "_AuthenticatedA2ABaseClient":
        if self._httpx_client is not None or self._client is not None:  # type: ignore[has-type]
            raise RuntimeError("A2ABaseClient already initialized")

        self._httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._task_timeout.total_seconds()),
        )

        if not self._agent_card:
            await self._resolve_agent_card()
        else:
            logger.info("Using pre-resolved agent card (registry lookup).")
        if not self._agent_card:
            raise RuntimeError("Agent card not resolved")

        # Allow auth providers that need agent-card parameters
        # (e.g. OAuth2CrossApplicationAccessOAuth2AuthProvider) to receive
        # the resolved card before the interceptor is set up.
        if isinstance(self._auth_provider, AgentCardAware):
            self._auth_provider.set_agent_card(self._agent_card)

        interceptors: list[Any] = []
        if self._auth_provider:
            if self._agent_card.security_schemes:
                # Agent card declares security schemes — use A2ACredentialService
                # for proper credential validation per the A2A spec.  This path
                # supports OAuth2 providers that need security-scheme negotiation.
                credential_service = A2ACredentialService(
                    auth_provider=self._auth_provider,
                    agent_card=self._agent_card,
                )
                interceptors.append(AuthInterceptor(credential_service))
                logger.info(
                    "Agent card declares security schemes, using security-scheme negotiation."
                )
            else:
                # No security schemes on the card — A2ACredentialService would
                # skip credential injection entirely.  Fall back to direct header
                # injection so simple auth providers (e.g. APIKeyAuthProvider)
                # still forward the token on every RPC call.
                user_id = Context.get().user_id or "default-user"
                auth_result = await self._auth_provider.authenticate(user_id=user_id)
                if auth_result:
                    assert self._httpx_client is not None
                    self._httpx_client.headers.update(_extract_auth_headers(auth_result))
                logger.info("No security schemes configured on the agent card, using default.")

        client_config = ClientConfig(
            httpx_client=self._httpx_client,
            streaming=self._streaming,
        )
        factory = ClientFactory(client_config)
        self._client = factory.create(self._agent_card, interceptors=interceptors)

        logger.info("Connected to A2A agent at %s", self._base_url)
        return self


class AgentCardRegistryLookup(BaseModel):
    """Identifies an agent card in the central DataRobot agent card registry.

    Exactly one of ``deployment_id`` or ``external_id`` must be specified.
    The registry is queried using standard DataRobot API-token authentication
    (``DATAROBOT_API_TOKEN``), which avoids the chicken-and-egg problem where the
    agent's own card endpoint requires per-agent AuthN/AuthZ.

    Example YAML::

        registry:
          deployment_id: "64a1b2c3d4e5f6a7b8c9d0e1"
    """

    deployment_id: str | None = Field(
        default=None,
        description="DataRobot deployment ID to look up in the central agent card registry.",
    )
    external_id: str | None = Field(
        default=None,
        description="External agent identifier to look up in the central agent card registry.",
    )

    @model_validator(mode="after")
    def _exactly_one_identifier(self) -> "AgentCardRegistryLookup":
        if self.deployment_id and self.external_id:
            raise ValueError(
                "Specify exactly one of 'deployment_id' or 'external_id' inside 'registry', "
                "not both. Each identifies the agent card differently in the central registry."
            )
        if not self.deployment_id and not self.external_id:
            raise ValueError(
                "The 'registry' block requires exactly one of 'deployment_id' or 'external_id' "
                "to identify the agent card in the central registry."
            )
        return self


class AuthenticatedA2AClientConfig(A2AClientConfig, name="authenticated_a2a_client"):  # type: ignore[call-arg]
    """A2A client config with separate discovery and call-phase auth.

    Supports two modes for agent card resolution:

    **Direct fetch** (existing behaviour) — set ``url`` to the agent's base URL
    and the card is fetched from ``{url}/.well-known/agent-card.json``::

        function_groups:
          my_agent:
            _type: authenticated_a2a_client
            url: "http://agent.example.com:8080"
            auth_provider: my_auth

    **Central registry lookup** — set ``registry`` with either ``deployment_id``
    or ``external_id``.  The card is fetched from the DataRobot central agent card
    registry using ``DATAROBOT_API_TOKEN``, and the agent's RPC URL is derived
    from the card's advertised ``url``::

        function_groups:
          my_agent:
            _type: authenticated_a2a_client
            registry:
              deployment_id: "64a1b2c3d4e5f6a7b8c9d0e1"
            auth_provider: okta_auth

    The two modes are mutually exclusive.
    """

    url: Any = Field(  # type: ignore[assignment]
        default=None,
        description="Base URL of the A2A agent for direct agent card fetch. "
        "Mutually exclusive with 'registry'.",
    )

    registry: AgentCardRegistryLookup | None = Field(
        default=None,
        description="Central DataRobot agent card registry lookup. Mutually exclusive with 'url'.",
    )

    @model_validator(mode="after")
    def _url_xor_registry(self) -> "AuthenticatedA2AClientConfig":
        has_url = self.url is not None
        has_registry = self.registry is not None
        if has_url and has_registry:
            raise ValueError(
                "Provide either 'url' for direct agent card fetch or 'registry' for "
                "central registry lookup, not both. "
                "When 'registry' is set, the agent's RPC URL is derived from the "
                "agent card's advertised URL."
            )
        if not has_url and not has_registry:
            raise ValueError(
                "Either 'url' or 'registry' must be provided. "
                "Use 'url' to fetch the agent card directly from the agent, "
                "or 'registry' to look it up in the central DataRobot agent card registry."
            )
        # Eager registration for batch prefetch (N+1 optimisation).
        # register() here is sync, no I/O — just queues the ID.
        # First async get() in __aenter__ flushes all pending IDs
        # in ≤2 HTTP calls; subsequent gets hit the warm cache.
        # See AgentCardRegistry docstring for full details.
        if has_registry:
            registry = get_default_registry_sync()
            registry.register(
                deployment_id=self.registry.deployment_id,  # type: ignore[union-attr]
                external_id=self.registry.external_id,  # type: ignore[union-attr]
            )
        return self


def _sanitize_a2a_error(exc: Exception) -> str:
    """Return a safe, single-line error description without sensitive material.

    Strips raw HTTP bodies, traceback detail, and anything that could echo
    back tokens or assertions.  Only the exception *type* and a curated
    category survive — the raw ``str(exc)`` is never surfaced.
    """
    # httpx status errors — include status + URL but NOT the response body
    # (which could echo submitted parameters like tokens or assertions).
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code} from {exc.request.url.host}{exc.request.url.path}"

    # Categorise by type so the LLM gets actionable context without
    # exposing the raw message which may contain sensitive material.
    if isinstance(exc, httpx.TimeoutException):
        return "request to remote agent timed out"

    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, ConnectionError, OSError)):
        return "network error communicating with remote agent"

    if isinstance(exc, (RuntimeError, ValueError)):
        return f"{type(exc).__name__}: authentication or protocol error"

    return f"{type(exc).__name__}: request to remote agent failed"


def _wrap_a2a_function(fn: Any) -> Any:
    """Wrap an A2A function so that exceptions are returned as error strings
    instead of propagating and crashing the agent.

    Works for both regular async functions and async generators.

    Sensitive material (tokens, assertions, HTTP bodies) is stripped from
    both the returned error string and the log line — only a sanitized
    summary is emitted.  The full traceback is deliberately **not** logged
    (``exc_info=False``) to prevent token values captured in frame locals
    from reaching log sinks.
    """
    # Check async generators first — they also satisfy iscoroutinefunction
    # in some Python versions, so the order matters.
    if inspect.isasyncgenfunction(fn):

        @functools.wraps(fn)
        async def _safe_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            try:
                async for event in fn(*args, **kwargs):
                    yield event
            except Exception as exc:
                safe_msg = _sanitize_a2a_error(exc)
                logger.error("A2A remote call failed: %s", safe_msg)
                logger.debug("A2A remote call exception detail: %s: %s", type(exc).__name__, exc)
                yield f"Error: failed to communicate with the remote agent: {safe_msg}"

        return _safe_gen

    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def _safe(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                safe_msg = _sanitize_a2a_error(exc)
                logger.error("A2A remote call failed: %s", safe_msg)
                logger.debug("A2A remote call exception detail: %s: %s", type(exc).__name__, exc)
                return f"Error: failed to communicate with the remote agent: {safe_msg}"

        return _safe

    return fn


class AuthenticatedA2AClientFunctionGroup(A2AClientFunctionGroup):
    """Uses :class:`_AuthenticatedA2ABaseClient` so both A2A phases are authenticated."""

    def add_function(self, name: str, fn: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """Intercept function registration to wrap *fn* with error handling."""
        super().add_function(name, _wrap_a2a_function(fn), **kwargs)

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

        # -------------------------------------------------------------------
        # Resolve agent card: registry lookup vs. direct fetch
        # -------------------------------------------------------------------
        pre_resolved_card: AgentCard | None = None

        if config.registry:
            # Fetch the card from the central DataRobot agent card registry
            registry = await get_default_registry()
            pre_resolved_card = await registry.get(
                deployment_id=config.registry.deployment_id,
                external_id=config.registry.external_id,
            )
            base_url = str(pre_resolved_card.url)
            logger.info(
                "Agent card resolved via central registry (deployment_id=%s, external_id=%s), "
                "RPC base URL derived from card: %s",
                config.registry.deployment_id,
                config.registry.external_id,
                base_url,
            )
        else:
            base_url = str(config.url)

        self._client = _AuthenticatedA2ABaseClient(
            base_url=base_url,
            agent_card_path=config.agent_card_path,
            task_timeout=config.task_timeout,
            streaming=config.streaming,
            auth_provider=auth_provider,
        )

        # Inject pre-resolved card so _AuthenticatedA2ABaseClient skips discovery
        if pre_resolved_card:
            self._client._agent_card = pre_resolved_card

        await self._client.__aenter__()

        logger.info(
            "Connected to A2A agent at %s (auth_provider: %s, user_id: %s, registry: %s)",
            base_url,
            config.auth_provider or "none",
            user_id,
            bool(config.registry),
        )

        self._register_functions()
        return self


@register_per_user_function_group(config_type=AuthenticatedA2AClientConfig)  # type: ignore[untyped-decorator]
async def authenticated_a2a_client(
    config: AuthenticatedA2AClientConfig, _builder: Builder
) -> AsyncGenerator[Any, None]:
    """NAT factory for :class:`AuthenticatedA2AClientFunctionGroup`."""
    async with AuthenticatedA2AClientFunctionGroup(config, _builder) as group:
        yield group
