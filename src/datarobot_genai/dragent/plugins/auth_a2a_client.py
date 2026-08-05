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
import mimetypes
from collections.abc import AsyncGenerator
from typing import Any
from typing import Protocol
from typing import runtime_checkable

import httpx
from a2a.client import A2ACardResolver
from a2a.client import AuthInterceptor
from a2a.client import ClientConfig
from a2a.client import ClientFactory
from a2a.types import AgentCapabilities
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

from datarobot_genai.dragent.a2a_artifact_client import OutboundFile
from datarobot_genai.dragent.a2a_artifact_client import build_client_message
from datarobot_genai.dragent.a2a_artifact_client import summarize_task
from datarobot_genai.dragent.agent_card_registry import AgentCardRegistryError
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

    async def send_parts(
        self,
        text: str | None = None,
        files: list[OutboundFile] | None = None,
        data: dict[str, Any] | None = None,
        *,
        task_id: str | None = None,
        context_id: str | None = None,
    ) -> list[Any]:
        """Send a multi-part message and return the raw response events.

        NAT's :meth:`send_message` takes ``message_text: str`` and builds a
        single ``TextPart``, so inbound files and structured data cannot be
        expressed through it. This sends a full ``Message`` instead.

        Authentication is unchanged: the message goes through the same a2a-sdk
        client and the same interceptors configured in :meth:`__aenter__`, so
        the Okta cross-application-access exchange applies exactly as it does to
        a text-only call.

        Args:
            text: Optional text body.
            files: Optional attachments (images, CSVs, PDFs ...).
            data: Optional structured payload.
            task_id: Set to continue an existing task.
            context_id: Set to stay within an existing conversation context.

        Returns:
            Raw response events. Each is either a ``Message`` or a
            ``ClientEvent`` tuple of ``(Task, UpdateEvent | None)``. Pass to
            :func:`~datarobot_genai.dragent.a2a_artifact_client.iter_artifacts`
            or :func:`~datarobot_genai.dragent.a2a_artifact_client.summarize_task`.

        Raises:
            RuntimeError: If the client was not initialised via ``async with``.
            ValueError: If no part of any kind was supplied.
        """
        if not self._client:
            raise RuntimeError(
                "A2A client not initialized -- enter the function group via "
                "'async with' before calling send_parts()."
            )

        message = build_client_message(
            text=text,
            files=files,
            data=data,
            task_id=task_id,
            context_id=context_id,
        )

        events: list[Any] = []
        async for event in self._client.send_message(message):
            events.append(event)

        logger.info(
            "A2A multi-part send returned %d event(s) (files=%d, data=%s)",
            len(events),
            len(files or []),
            data is not None,
        )
        return events


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

    artifact_client: bool = Field(
        default=False,
        description="Register artifact-aware functions alongside NAT's text-only "
        "'call'. Adds 'send_with_attachments' (send files and structured data, "
        "read every returned artifact) and 'get_task_artifacts' (re-read a known "
        "task's artifacts). Off by default: registering extra functions changes "
        "which tools an agent's LLM can select, so this is opt-in.",
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

    # Safe to surface verbatim — crafted by this codebase, never contains secrets.
    # Checked before RuntimeError (its superclass).
    if isinstance(exc, AgentCardRegistryError):
        return f"agent card registry error: {exc}"

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


class _FailedRegistryClient:
    """Minimal stand-in providing ``agent_card`` for ``_register_functions()``.

    Used when registry lookup fails.  Only ``agent_card`` and ``__aexit__``
    are needed — the actual RPC methods are never called because
    ``add_function`` replaces every registered function before it can
    reference the client.
    """

    __slots__ = ("_agent_card",)

    def __init__(self, agent_card: AgentCard) -> None:
        self._agent_card = agent_card

    @property
    def agent_card(self) -> AgentCard:
        return self._agent_card

    async def __aexit__(self, *args: Any) -> None:
        """No-op — nothing to close."""


def _make_placeholder_agent_card(error_msg: str) -> AgentCard:
    """Build a minimal ``AgentCard`` for ``_register_functions()`` to read.

    The description embeds the original error for observability in logs.
    """
    return AgentCard(
        name="unavailable",
        description=f"Agent card could not be resolved: {error_msg}",
        url="https://unavailable/",
        version="0.0.0",
        skills=[],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
    )


class AuthenticatedA2AClientFunctionGroup(A2AClientFunctionGroup):
    """Uses :class:`_AuthenticatedA2ABaseClient` so both A2A phases are authenticated."""

    def add_function(self, name: str, fn: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """Intercept function registration to wrap *fn* with error handling.

        In degraded mode (registry lookup failed), replaces *fn* entirely
        with one that raises the stored error — so ``_wrap_a2a_function``
        catches it and returns an actionable message to the LLM.
        This avoids coupling to NAT's client method signatures or protocols.
        """
        registry_error = getattr(self, "_registry_error", None)
        if registry_error is not None:
            err = registry_error

            @functools.wraps(fn)
            async def _raise_registry_error(*args: Any, **kwargs: Any) -> Any:
                raise err

            fn = _raise_registry_error
        super().add_function(name, _wrap_a2a_function(fn), **kwargs)

    async def send_parts(
        self,
        text: str | None = None,
        files: list[OutboundFile] | None = None,
        data: dict[str, Any] | None = None,
        *,
        task_id: str | None = None,
        context_id: str | None = None,
    ) -> list[Any]:
        """Send a multi-part A2A message and return the raw response events.

        The programmatic entry point for callers that need files, images or
        structured data on the wire. Authentication -- including the Okta
        cross-application-access exchange -- applies unchanged, because this
        reuses the function group's own authenticated client.

        Args:
            text: Optional text body.
            files: Optional attachments.
            data: Optional structured payload.
            task_id: Set to continue an existing task.
            context_id: Set to stay within an existing conversation context.

        Returns:
            Raw response events, for
            :func:`~datarobot_genai.dragent.a2a_artifact_client.iter_artifacts`,
            :func:`~datarobot_genai.dragent.a2a_artifact_client.summarize_task`
            or
            :func:`~datarobot_genai.dragent.a2a_artifact_client.save_task_files`.

        Raises:
            RuntimeError: If the client is unavailable, including when a registry
                lookup failed and the group initialised in degraded mode.
            ValueError: If no part of any kind was supplied.
        """
        registry_error = getattr(self, "_registry_error", None)
        if registry_error is not None:
            raise RuntimeError(
                f"A2A client unavailable: agent card registry lookup failed ({registry_error})"
            )

        send_parts = getattr(self._client, "send_parts", None)
        if send_parts is None:
            raise RuntimeError(
                "A2A client does not support multi-part send -- expected "
                "_AuthenticatedA2ABaseClient. Ensure the function group was "
                "entered via 'async with'."
            )

        return await send_parts(  # type: ignore[no-any-return]
            text=text,
            files=files,
            data=data,
            task_id=task_id,
            context_id=context_id,
        )

    def _register_functions(self) -> None:
        """Register NAT's functions, plus artifact-aware ones when enabled.

        ``super()`` registers the stock three-level API, of which ``call`` is
        text-only in both directions. When ``artifact_client`` is set, two more
        functions are added so an agent's LLM can send attachments and read
        artifacts without the application writing its own A2A client.
        """
        super()._register_functions()

        config: AuthenticatedA2AClientConfig = self._config  # type: ignore[assignment]
        if not getattr(config, "artifact_client", False):
            return

        async def send_with_attachments(
            message: str,
            attach_data: dict[str, Any] | None = None,
            attach_uris: list[str] | None = None,
        ) -> str:
            """Send a message to the remote agent and report every artifact returned.

            Files are attached **by URI** rather than by content, because a model
            cannot emit raw bytes. To send bytes, call
            :meth:`AuthenticatedA2AClientFunctionGroup.send_parts` from code with
            :class:`~datarobot_genai.dragent.a2a_artifact_client.OutboundFile`.

            Args:
                message: The text to send.
                attach_data: Optional structured payload, sent as a DataPart.
                attach_uris: Optional file URIs, each sent as a FilePart. The MIME
                    type is inferred from the extension. Note that an A2A URI
                    carries none of the call's authentication, so the receiving
                    agent must be able to fetch it independently.

            Returns:
                A rendered report of the task state and every artifact.
            """
            files = [
                OutboundFile(
                    name=uri.rstrip("/").rsplit("/", 1)[-1] or "attachment",
                    mime_type=mimetypes.guess_type(uri)[0] or "application/octet-stream",
                    uri=uri,
                )
                for uri in attach_uris or []
            ]
            events = await self.send_parts(
                text=message, files=files or None, data=attach_data
            )
            return summarize_task(events)

        async def get_task_artifacts(task_id: str) -> str:
            """Re-read the artifacts of a task by id.

            Args:
                task_id: The task to inspect.

            Returns:
                A rendered report of the task state and every artifact.
            """
            task = await self._client.get_task(task_id)  # type: ignore[union-attr]
            return summarize_task([task])

        self.add_function(
            name="send_with_attachments",
            fn=send_with_attachments,
            description=(
                "Send a message to this agent and read back everything it returns, "
                "including files, images and structured data. Use this instead of "
                "'call' whenever the reply may contain artifacts, or when the user "
                "asks what files or images the other agent produced."
            ),
        )
        self.add_function(
            name="get_task_artifacts",
            fn=get_task_artifacts,
            description=(
                "List the artifacts (files, images, structured data) attached to a "
                "known task_id on this agent."
            ),
        )
        logger.info(
            "Registered artifact-aware A2A client functions "
            "(send_with_attachments, get_task_artifacts)"
        )

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
            # Catch AgentCardRegistryError so the function group initialises
            # in a degraded state instead of crashing (generic JSON-RPC -32603).
            try:
                registry = await get_default_registry()
                pre_resolved_card = await registry.get(
                    deployment_id=config.registry.deployment_id,
                    external_id=config.registry.external_id,
                )
            except AgentCardRegistryError as exc:
                error_msg = str(exc)
                logger.error(
                    "Agent card registry lookup failed: %s",
                    error_msg,
                )
                # Store the error so add_function replaces every registered
                # function with one that raises it (caught by _wrap_a2a_function).
                self._registry_error = exc
                placeholder_card = _make_placeholder_agent_card(error_msg)
                self._client = _FailedRegistryClient(placeholder_card)  # type: ignore[assignment]
                self._register_functions()
                return self

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
