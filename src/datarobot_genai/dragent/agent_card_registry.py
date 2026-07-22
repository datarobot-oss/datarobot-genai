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

"""Client for the central DataRobot agent card registry.

The central registry provides a tenant-scoped list of agent cards that requires
only standard DataRobot API-token authentication (``DATAROBOT_API_TOKEN``).
This avoids the chicken-and-egg problem where an individual agent's card
endpoint is behind per-agent AuthN/AuthZ.

The :class:`AgentCardRegistry` supports **batch fetching** so that many
function groups sharing the same workflow can resolve all their cards in a
minimum number of HTTP round-trips instead of N+1 individual requests.

Lookups can be **registered** before the first fetch so that ``get()``
automatically triggers a single batch prefetch for all pending IDs on its
first invocation — no explicit ``prefetch()`` call required.

.. note::
    The registry API uses AND semantics when both ``deploymentIds`` and
    ``externalIds`` parameters are provided in a single request, which would
    return empty results.  Therefore the registry issues **separate** HTTP
    calls for deployment IDs and external IDs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from typing import Literal
from typing import NamedTuple

import httpx
from a2a.types import AgentCard
from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from pydantic import Field

from datarobot_genai.dragent.agent_card_registry_backends import AgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import LayeredAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import LookupKeyType
from datarobot_genai.dragent.agent_card_registry_backends import MemoryAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import create_agent_card_cache_backend
from datarobot_genai.dragent.deployment_urls import build_agent_cards_registry_url

logger = logging.getLogger(__name__)

# Default cache TTL: 24 hours (in seconds).
_DEFAULT_CACHE_TTL_SECONDS = 24 * 3600

# Default hard staleness bound for stale-if-error (in seconds).
_DEFAULT_MAX_STALENESS_SECONDS = 24 * 3600

# Default background refresh interval (in seconds). 0 disables the refresh loop.
_DEFAULT_REFRESH_INTERVAL_SECONDS = 30 * 60

# Default HTTP timeout for registry requests (in seconds).
_DEFAULT_TIMEOUT_SECONDS = 30.0

# Maximum page size accepted by the registry API.
_MAX_PAGE_SIZE = 100

# Safety limit to prevent infinite pagination loops.
_MAX_PAGES = 100

# Allowed strategies for duplicate external IDs.
DuplicateStrategy = Literal["first", "last", "error"]


class DataRobotRegistrySettings(DataRobotAppFrameworkBaseSettings):
    """DataRobot connection settings for the central agent card registry.

    Loads ``DATAROBOT_API_TOKEN`` and ``DATAROBOT_ENDPOINT`` from env vars
    (including Runtime Parameters), ``.env``, file secrets, or Pulumi config
    using the standard :class:`DataRobotAppFrameworkBaseSettings` priority
    chain.
    """

    datarobot_api_token: str | None = None
    datarobot_endpoint: str | None = None


class AgentCardRegistryConfig(DataRobotAppFrameworkBaseSettings):
    """Configuration for the agent card registry cache.

    Controllable via environment variables (prefix-free, following the
    standard :class:`DataRobotAppFrameworkBaseSettings` resolution chain).

    Set ``AGENT_CARD_REGISTRY_CACHE_TTL=0`` to disable caching entirely
    (every ``get()`` triggers a fresh HTTP fetch).
    """

    agent_card_registry_cache_ttl: int = Field(
        default=_DEFAULT_CACHE_TTL_SECONDS,
        ge=0,
        description=(
            "Time-to-live for cached agent cards in seconds. "
            "Set to 0 to disable caching (every get() triggers a fresh fetch). "
            "Default: 86400 (24 hours)."
        ),
    )

    agent_card_registry_timeout: float = Field(
        default=_DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        description="HTTP timeout in seconds for registry API requests. Default: 30.",
    )

    agent_card_registry_on_duplicate: DuplicateStrategy = Field(
        default="first",
        description=(
            "Strategy when the registry returns multiple agent cards for the "
            "same external ID.  The registry API returns cards sorted by "
            "creation time (ascending), so 'first' keeps the earliest "
            "registered card, 'last' keeps the most recently registered card, "
            "and 'error' raises AgentCardRegistryError.  Default: 'first'."
        ),
    )

    agent_card_registry_prefetch_on_startup: bool = Field(
        default=True,
        description=(
            "When true, batch-fetch all registry-backed agent cards during "
            "dragent FastAPI startup (before accepting traffic). "
            "Set AGENT_CARD_REGISTRY_PREFETCH_ON_STARTUP=false to disable."
        ),
    )

    agent_card_registry_max_staleness_seconds: int = Field(
        default=_DEFAULT_MAX_STALENESS_SECONDS,
        ge=0,
        description=(
            "Maximum age in seconds for serving a cached agent card when the "
            "registry is unreachable (stale-if-error). Default: 86400 (24 hours)."
        ),
    )

    agent_card_registry_stale_if_error: bool = Field(
        default=True,
        description=(
            "When true, return the last-known-good cached agent card if a "
            "registry fetch fails and the entry is within "
            "agent_card_registry_max_staleness_seconds. "
            "Set AGENT_CARD_REGISTRY_STALE_IF_ERROR=false to disable."
        ),
    )

    agent_card_registry_backend: Literal["memory", "redis"] = Field(
        default="memory",
        description=(
            "Cache backend for agent cards. 'memory' uses in-process cache only; "
            "'redis' adds a shared Redis L2 with in-process L1 read-through."
        ),
    )

    agent_card_registry_redis_url: str | None = Field(
        default=None,
        description="Redis URL (required when agent_card_registry_backend='redis').",
    )

    agent_card_registry_redis_prefix: str = Field(
        default="dragent:",
        description="Key prefix for Redis agent card cache entries.",
    )

    agent_card_registry_refresh_interval_seconds: int = Field(
        default=_DEFAULT_REFRESH_INTERVAL_SECONDS,
        ge=0,
        description=(
            "Background refresh period in seconds for registered agent cards "
            "past the soft cache TTL. Set to 0 to disable. Default: 1800 (30 min)."
        ),
    )


class AgentCardRegistryError(RuntimeError):
    """Raised when the central agent card registry lookup fails."""


def _resolve_settings(
    api_token: str | None = None,
    endpoint: str | None = None,
) -> tuple[str, str]:
    """Return validated ``(api_token, endpoint)`` from explicit values or settings."""
    settings = DataRobotRegistrySettings()
    resolved_token = api_token or settings.datarobot_api_token
    if not resolved_token:
        raise AgentCardRegistryError(
            "DataRobot API token is required for agent card registry lookup. "
            "Set the DATAROBOT_API_TOKEN environment variable or provide it explicitly."
        )
    resolved_endpoint = endpoint or settings.datarobot_endpoint
    if not resolved_endpoint:
        raise AgentCardRegistryError(
            "DataRobot API endpoint is required for agent card registry lookup. "
            "Set the DATAROBOT_ENDPOINT environment variable or provide it explicitly."
        )
    return resolved_token, resolved_endpoint


class ParsedRegistryCards(NamedTuple):
    """Parsed registry response with lookup key types for cache indexing."""

    cards: dict[str, AgentCard]
    key_types: dict[str, LookupKeyType]


def _parse_registry_response(
    body: dict[str, Any],
    on_duplicate: DuplicateStrategy = "first",
) -> ParsedRegistryCards:
    """Parse a paginated registry response into ``{id: AgentCard}``.

    Each record is indexed by ``deploymentId`` (always unique) and
    ``externalId`` (may have duplicates).  The ``on_duplicate`` strategy
    controls what happens when multiple entries share the same external ID.

    The registry API returns entries sorted by ``_id`` ascending (creation
    time), so the iteration order matches chronological registration order:

    * ``"first"`` — keep the earliest registered card, log a warning.
    * ``"last"`` — keep the most recently registered card, log a warning.
    * ``"error"`` — raise :class:`AgentCardRegistryError`.
    """
    cards: dict[str, AgentCard] = {}
    key_types: dict[str, LookupKeyType] = {}
    for entry in body.get("data", []):
        raw_card = entry.get("agentCard")
        if not raw_card:
            logger.warning(
                "Registry entry %s has no 'agentCard' payload — skipping.",
                entry.get("id", "?"),
            )
            continue
        try:
            card = AgentCard.model_validate(raw_card)
        except Exception:
            logger.warning(
                "Failed to parse agent card for registry entry %s — skipping.",
                entry.get("id", "?"),
                exc_info=True,
            )
            continue

        # Deployment IDs are unique by platform design — always overwrite.
        if dep_id := entry.get("deploymentId"):
            cards[dep_id] = card
            key_types[dep_id] = "dep"

        # External IDs may have duplicates — apply the configured strategy.
        if ext_id := entry.get("externalId"):
            if ext_id not in cards:
                cards[ext_id] = card
                key_types[ext_id] = "ext"
            else:
                logger.warning(
                    "Duplicate external ID '%s' in registry response (on_duplicate=%s).",
                    ext_id,
                    on_duplicate,
                )
                if on_duplicate == "error":
                    raise AgentCardRegistryError(
                        f"Multiple agent cards found for external_id='{ext_id}'. "
                        "Set AGENT_CARD_REGISTRY_ON_DUPLICATE='first' or 'last' "
                        "to pick one, or fix the duplicate registrations."
                    )
                if on_duplicate == "last":
                    cards[ext_id] = card
                    key_types[ext_id] = "ext"
                # "first" — keep existing entry (no-op)
    return ParsedRegistryCards(cards=cards, key_types=key_types)


class AgentCardRegistry:
    """Batch-capable, TTL-cached client for the central agent card registry.

    IDs are ``register()``-ed synchronously at config-parse time (no I/O).
    The first ``get()`` flushes all pending IDs in ≤2 HTTP calls
    (one per ID type — API uses AND when both are mixed).
    Subsequent ``get()`` calls hit the in-memory cache until the soft TTL
    (``AGENT_CARD_REGISTRY_CACHE_TTL``) expires.  When a refresh fails and
    ``AGENT_CARD_REGISTRY_STALE_IF_ERROR`` is enabled, a cached card may still
    be returned up to ``AGENT_CARD_REGISTRY_MAX_STALENESS_SECONDS``.
    """

    def __init__(
        self,
        api_token: str | None = None,
        endpoint: str | None = None,
        timeout: float | None = None,
        cache_ttl: int | None = None,
        max_staleness_seconds: int | None = None,
        stale_if_error: bool | None = None,
        on_duplicate: DuplicateStrategy | None = None,
        cache_backend: AgentCardCacheBackend | None = None,
    ) -> None:
        self._api_token = api_token
        self._endpoint = endpoint
        self._lock = asyncio.Lock()

        # Pending registrations (filled synchronously, flushed on first get)
        self._pending_deployment_ids: set[str] = set()
        self._pending_external_ids: set[str] = set()

        # All IDs registered at config-parse time (used for background refresh)
        self._registered_deployment_ids: set[str] = set()
        self._registered_external_ids: set[str] = set()

        config = AgentCardRegistryConfig()
        self._timeout = timeout if timeout is not None else config.agent_card_registry_timeout
        self._cache_ttl = (
            cache_ttl if cache_ttl is not None else config.agent_card_registry_cache_ttl
        )
        self._max_staleness_seconds = (
            max_staleness_seconds
            if max_staleness_seconds is not None
            else config.agent_card_registry_max_staleness_seconds
        )
        self._stale_if_error = (
            stale_if_error
            if stale_if_error is not None
            else config.agent_card_registry_stale_if_error
        )
        self._on_duplicate: DuplicateStrategy = (
            on_duplicate if on_duplicate is not None else config.agent_card_registry_on_duplicate
        )
        self._backend = cache_backend or create_agent_card_cache_backend(config)

        logger.debug(
            "AgentCardRegistry created (backend=%s, cache_ttl=%ds, max_staleness=%ds, "
            "stale_if_error=%s)",
            config.agent_card_registry_backend,
            self._cache_ttl,
            self._max_staleness_seconds,
            self._stale_if_error,
        )

    # ------------------------------------------------------------------
    # Registration (synchronous — called at config-parse time)
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        deployment_id: str | None = None,
        external_id: str | None = None,
    ) -> None:
        """Declare intent to look up an agent card.

        Call this at config-parse/validation time so that the first
        :meth:`get` can batch all pending IDs into a single prefetch.

        Exactly one of ``deployment_id`` or ``external_id`` must be given.
        """
        if deployment_id:
            self._pending_deployment_ids.add(deployment_id)
            self._registered_deployment_ids.add(deployment_id)
        elif external_id:
            self._pending_external_ids.add(external_id)
            self._registered_external_ids.add(external_id)

    def has_registered_lookups(self) -> bool:
        """Return whether any registry lookup IDs were registered."""
        return bool(self._registered_deployment_ids or self._registered_external_ids)

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    async def _fetch(self, params: dict[str, str]) -> ParsedRegistryCards:
        """Execute registry HTTP GET(s) with pagination and return all parsed cards.

        Requests the maximum page size (100) to minimise round-trips, then
        follows ``next`` links until all pages are consumed.  All entries are
        accumulated and parsed together so that the ``on_duplicate`` strategy
        is applied consistently across the full result set.
        """
        token, endpoint = _resolve_settings(self._api_token, self._endpoint)
        registry_url = build_agent_cards_registry_url(endpoint)
        headers = {"Authorization": f"Bearer {token}"}
        params_with_limit = {"limit": str(_MAX_PAGE_SIZE), **params}

        logger.info(
            "Fetching agent cards from registry: %s (params=%s)",
            registry_url,
            params,
        )

        all_entries: list[dict[str, Any]] = []

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
                response = await client.get(registry_url, params=params_with_limit, headers=headers)
                response.raise_for_status()
                body = response.json()
                all_entries.extend(body.get("data", []))

                next_url = body.get("next")
                pages_fetched = 1
                while next_url:
                    if pages_fetched >= _MAX_PAGES:
                        logger.warning(
                            "Pagination safety limit reached (%d pages). "
                            "Some agent cards may not have been fetched.",
                            _MAX_PAGES,
                        )
                        break
                    logger.debug("Following pagination link (page %d).", pages_fetched + 1)
                    response = await client.get(next_url, headers=headers)
                    response.raise_for_status()
                    body = response.json()
                    all_entries.extend(body.get("data", []))
                    next_url = body.get("next")
                    pages_fetched += 1

        except httpx.HTTPStatusError as exc:
            raise AgentCardRegistryError(
                f"Agent card registry request failed with HTTP "
                f"{exc.response.status_code}. Verify your API token and that "
                f"the agents are registered."
            ) from exc
        except httpx.HTTPError as exc:
            raise AgentCardRegistryError(f"Agent card registry request failed: {exc}") from exc

        parsed = _parse_registry_response({"data": all_entries}, on_duplicate=self._on_duplicate)
        logger.info(
            "Fetched %d agent card(s) from registry (%d pages).",
            len(parsed.cards),
            pages_fetched,
        )
        return parsed

    async def _is_fresh(self, key: str) -> bool:
        """Return True if *key* is cached and within the soft TTL."""
        record = await self._backend.get_fresh(key, cache_ttl=self._cache_ttl)
        return record is not None

    async def _try_get_stale(self, key: str) -> AgentCard | None:
        """Return a stale cached card when refresh failed and policy allows it."""
        if not self._stale_if_error:
            return None
        record = await self._backend.get_stale(
            key,
            max_staleness_seconds=self._max_staleness_seconds,
        )
        if record is None:
            return None
        logger.warning(
            "Registry unreachable; serving stale agent card for %s (age=%.0fs)",
            key,
            record.age_seconds(),
        )
        return record.card

    async def _store_cards(self, parsed: ParsedRegistryCards) -> None:
        if not parsed.cards:
            return
        await self._backend.store(parsed.cards, key_types=parsed.key_types)

    def _age_cache_entry_for_test(self, lookup_key: str, seconds: float) -> None:
        """Shift a cached entry's fetch time backward (tests only)."""
        backend = self._backend
        if isinstance(backend, MemoryAgentCardCacheBackend):
            backend.age_entry_for_test(lookup_key, seconds)
            return

        if isinstance(backend, LayeredAgentCardCacheBackend):
            backend.memory.age_entry_for_test(lookup_key, seconds)

    async def _flush_pending(self) -> None:
        """Batch-fetch all registered-but-uncached IDs.  Must be called under ``_lock``."""
        missing_dep = [d for d in self._pending_deployment_ids if not await self._is_fresh(d)]
        missing_ext = [e for e in self._pending_external_ids if not await self._is_fresh(e)]

        # Clear pending sets regardless — they've been processed
        self._pending_deployment_ids.clear()
        self._pending_external_ids.clear()

        if not missing_dep and not missing_ext:
            return

        if missing_dep:
            parsed = await self._fetch({"deploymentIds": ",".join(missing_dep)})
            await self._store_cards(parsed)

        if missing_ext:
            parsed = await self._fetch({"externalIds": ",".join(missing_ext)})
            await self._store_cards(parsed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def prefetch(
        self,
        *,
        deployment_ids: list[str] | None = None,
        external_ids: list[str] | None = None,
    ) -> None:
        """Batch-fetch and cache agent cards in a minimum number of HTTP calls.

        Issues **separate** requests for ``deployment_ids`` and
        ``external_ids`` because the API uses AND semantics when both
        parameters are present in a single request.

        Already-cached (non-expired) IDs are skipped.

        Parameters
        ----------
        deployment_ids:
            Deployment IDs to prefetch.
        external_ids:
            External IDs to prefetch.
        """
        async with self._lock:
            missing_dep = [d for d in (deployment_ids or []) if not await self._is_fresh(d)]
            missing_ext = [e for e in (external_ids or []) if not await self._is_fresh(e)]

            if not missing_dep and not missing_ext:
                logger.debug("All requested agent cards already cached — skipping prefetch.")
                return

            if missing_dep:
                parsed = await self._fetch({"deploymentIds": ",".join(missing_dep)})
                await self._store_cards(parsed)

            if missing_ext:
                parsed = await self._fetch({"externalIds": ",".join(missing_ext)})
                await self._store_cards(parsed)

    async def refresh_all_registered(self) -> None:
        """Re-fetch registered IDs whose cache entries are past the soft TTL.

        Failures are logged and existing cache entries are left in place so
        stale-if-error can continue serving them during registry outages.
        """
        if not self._registered_deployment_ids and not self._registered_external_ids:
            logger.debug("No registered agent card IDs; skipping background refresh.")
            return

        deployment_ids = sorted(self._registered_deployment_ids)
        external_ids = sorted(self._registered_external_ids)
        logger.debug(
            "Refreshing registered agent cards (deployment_ids=%s, external_ids=%s)",
            deployment_ids,
            external_ids,
        )
        try:
            await self.prefetch(
                deployment_ids=deployment_ids or None,
                external_ids=external_ids or None,
            )
        except AgentCardRegistryError:
            logger.warning(
                "Background agent card registry refresh failed; keeping cached entries.",
                exc_info=True,
            )

    async def get(
        self,
        *,
        deployment_id: str | None = None,
        external_id: str | None = None,
    ) -> AgentCard:
        """Return a single agent card, using the cache or fetching on demand.

        On the first call, all IDs previously passed to :meth:`register`
        are batch-fetched in a single prefetch (at most 2 HTTP calls).
        Subsequent calls for already-cached, non-expired cards are instant.

        Exactly one of ``deployment_id`` or ``external_id`` must be provided.

        Raises
        ------
        AgentCardRegistryError
            If the card cannot be found or the request fails.
        """
        if bool(deployment_id) == bool(external_id):
            raise AgentCardRegistryError("Specify exactly one of 'deployment_id' or 'external_id'.")

        lookup_key: str = deployment_id or external_id  # type: ignore[assignment]

        # Fast path — fresh cache hit
        if fresh := await self._backend.get_fresh(lookup_key, cache_ttl=self._cache_ttl):
            return fresh.card

        async with self._lock:
            # Double-check after acquiring lock
            if fresh := await self._backend.get_fresh(lookup_key, cache_ttl=self._cache_ttl):
                return fresh.card

            try:
                # Flush all pending registrations in a batch
                if self._pending_deployment_ids or self._pending_external_ids:
                    await self._flush_pending()
                    if fresh := await self._backend.get_fresh(
                        lookup_key, cache_ttl=self._cache_ttl
                    ):
                        return fresh.card

                # Still not fresh — fetch individually
                params: dict[str, str] = (
                    {"deploymentIds": deployment_id}
                    if deployment_id
                    else {"externalIds": external_id}  # type: ignore[dict-item]
                )
                parsed = await self._fetch(params)
                await self._store_cards(parsed)

                if fresh := await self._backend.get_fresh(lookup_key, cache_ttl=self._cache_ttl):
                    return fresh.card
            except AgentCardRegistryError:
                if stale_card := await self._try_get_stale(lookup_key):
                    return stale_card
                raise

        # Fetch succeeded but the requested key was absent from the response.
        id_label = (
            f"deployment_id='{deployment_id}'" if deployment_id else f"external_id='{external_id}'"
        )
        raise AgentCardRegistryError(
            f"No agent card found in the central registry for {id_label}. "
            "Verify that the deployment exists and is registered in your organisation."
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class _RegistryHolder:
    """Mutable container for the singleton."""

    instance: AgentCardRegistry | None = None
    lock = asyncio.Lock()


async def get_default_registry() -> AgentCardRegistry:
    """Return the module-level :class:`AgentCardRegistry` singleton.

    Created lazily on first access.  Credentials are resolved from
    :class:`DataRobotRegistrySettings` at instantiation time.
    """
    if _RegistryHolder.instance is None:
        async with _RegistryHolder.lock:
            if _RegistryHolder.instance is None:
                _RegistryHolder.instance = AgentCardRegistry()
    return _RegistryHolder.instance


def get_default_registry_sync() -> AgentCardRegistry:
    """Return the singleton, creating it if needed (synchronous).

    Safe to call from pydantic validators and other sync contexts
    (e.g. config-parse time) because :class:`AgentCardRegistry.__init__`
    does no I/O.
    """
    if _RegistryHolder.instance is None:
        _RegistryHolder.instance = AgentCardRegistry()
    return _RegistryHolder.instance


def reset_default_registry() -> None:
    """Reset the module-level singleton (for testing)."""
    _RegistryHolder.instance = None
