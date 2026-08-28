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
    The registry issues **one HTTP call per ID kind** — deployment, external
    and workload IDs are never mixed in a single request.  Sending
    ``deploymentIds`` together with ``workloadIds`` is rejected by the API with
    HTTP 400 (the two are mutually exclusive), and combining either with
    ``externalIds`` AND-matches, returning empty results.

.. note::
    The API caps each ID parameter at 20 values, so :meth:`AgentCardRegistry._fetch`
    splits longer ID lists into chunks of 20 and accumulates the entries before
    parsing them as one result set.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from collections.abc import Iterator
from typing import Any
from typing import Literal
from typing import NamedTuple

import httpx
from a2a.types import AgentCard
from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from pydantic import Field

from datarobot_genai.core.config import resolve_config
from datarobot_genai.dragent.agent_card_registry_backends import AgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import LayeredAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import LookupKeyType
from datarobot_genai.dragent.agent_card_registry_backends import MemoryAgentCardCacheBackend
from datarobot_genai.dragent.agent_card_registry_backends import RegistryIds
from datarobot_genai.dragent.agent_card_registry_backends import create_agent_card_cache_backend
from datarobot_genai.dragent.deployment_urls import build_agent_cards_registry_url

logger = logging.getLogger(__name__)

# Default cache TTL: 24 hours (in seconds).
_DEFAULT_CACHE_TTL_SECONDS = 24 * 3600

# Default HTTP timeout for registry requests (in seconds).
_DEFAULT_TIMEOUT_SECONDS = 30.0

# Maximum page size accepted by the registry API.
_MAX_PAGE_SIZE = 100

# Maximum number of IDs the registry API accepts in a single ID parameter.
_MAX_IDS_PER_REQUEST = 20

# Registry query parameter per lookup key type.
_DEPLOYMENT_IDS_PARAM = "deploymentIds"
_EXTERNAL_IDS_PARAM = "externalIds"
_WORKLOAD_IDS_PARAM = "workloadIds"
_ID_PARAMS = (_DEPLOYMENT_IDS_PARAM, _EXTERNAL_IDS_PARAM, _WORKLOAD_IDS_PARAM)

# Safety limit to prevent infinite pagination loops.
_MAX_PAGES = 100

# Allowed strategies for duplicate external IDs.
DuplicateStrategy = Literal["first", "last", "error"]


class DataRobotRegistrySettings(DataRobotAppFrameworkBaseSettings):
    """DataRobot endpoint setting for the central agent card registry.

    Loads ``DATAROBOT_ENDPOINT`` from env vars (including Runtime Parameters),
    ``.env``, file secrets, or Pulumi config using the standard
    :class:`DataRobotAppFrameworkBaseSettings` priority chain.

    Deliberately *not* read through ``resolve_config().resolve_datarobot_endpoint()``:
    that resolver substitutes the public ``app.datarobot.com`` default when nothing
    is configured, and this lookup sends the API token to whatever host it resolves,
    so an unset endpoint has to stay distinguishable from a configured one. The API
    token itself does come from the global config; see :func:`_resolve_settings`.
    """

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

    agent_card_registry_memory_space_id: str | None = Field(
        default=None,
        description=(
            "DataRobot MemorySpace ID for the agent card registry L2 cache. "
            "When unset, only in-process L1 caching is used."
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
    resolved_token = api_token or resolve_config().resolve_datarobot_api_token()
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


def _id_param_for(key_type: LookupKeyType) -> str:
    """Return the registry query parameter that looks up *key_type* IDs."""
    return {
        "deployment": _DEPLOYMENT_IDS_PARAM,
        "external": _EXTERNAL_IDS_PARAM,
        "workload": _WORKLOAD_IDS_PARAM,
    }[key_type]


def _chunk_id_params(params: dict[str, str]) -> Iterator[dict[str, str]]:
    """Split *params* into requests that respect the API's per-parameter ID cap.

    Yields *params* unchanged when no ID parameter exceeds
    :data:`_MAX_IDS_PER_REQUEST`; otherwise yields one dict per chunk of IDs.
    Only one ID parameter is ever present per request (callers issue one call
    per ID kind), so chunking never has to combine chunks across parameters.

    Raises
    ------
    AgentCardRegistryError
        If ``deploymentIds`` and ``workloadIds`` are both present.  The API
        rejects that combination with HTTP 400, so this catches the mistake at
        the call site instead of as an opaque request failure.
    """
    if _DEPLOYMENT_IDS_PARAM in params and _WORKLOAD_IDS_PARAM in params:
        raise AgentCardRegistryError(
            f"Cannot request '{_DEPLOYMENT_IDS_PARAM}' and '{_WORKLOAD_IDS_PARAM}' in the "
            "same agent card registry request — the API rejects the combination. "
            "Issue one request per ID kind."
        )

    id_params = [key for key in _ID_PARAMS if key in params]
    for key in id_params:
        ids = params[key].split(",")
        if len(ids) <= _MAX_IDS_PER_REQUEST:
            continue
        other = {k: v for k, v in params.items() if k != key}
        for start in range(0, len(ids), _MAX_IDS_PER_REQUEST):
            chunk = ids[start : start + _MAX_IDS_PER_REQUEST]
            yield {**other, key: ",".join(chunk)}
        return

    yield params


class ParsedRegistryCards(NamedTuple):
    """Parsed registry response with lookup key types for cache indexing."""

    cards: dict[str, AgentCard]
    key_types: dict[str, LookupKeyType]
    registry_ids: dict[str, RegistryIds]


def _parse_registry_response(
    body: dict[str, Any],
    on_duplicate: DuplicateStrategy = "first",
) -> ParsedRegistryCards:
    """Parse a paginated registry response into ``{id: AgentCard}``.

    Each record is indexed by its primary key — ``deploymentId`` or
    ``workloadId``, both unique by platform design — and, independently, by
    ``externalId`` (which may have duplicates).  A workload-hosted agent that
    also publishes an external ID is therefore reachable by either.  The
    ``on_duplicate`` strategy controls what happens when multiple entries share
    the same external ID.

    The registry API returns entries sorted by ``_id`` ascending (creation
    time), so the iteration order matches chronological registration order:

    * ``"first"`` — keep the earliest registered card, log a warning.
    * ``"last"`` — keep the most recently registered card, log a warning.
    * ``"error"`` — raise :class:`AgentCardRegistryError`.
    """
    cards: dict[str, AgentCard] = {}
    key_types: dict[str, LookupKeyType] = {}
    registry_ids: dict[str, RegistryIds] = {}
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

        dep_id = entry.get("deploymentId")
        ext_id = entry.get("externalId")
        wl_id = entry.get("workloadId")
        ids = RegistryIds(deployment_id=dep_id, external_id=ext_id, workload_id=wl_id)

        # Deployment IDs are unique by platform design — always overwrite.
        if dep_id:
            cards[dep_id] = card
            key_types[dep_id] = "deployment"
            registry_ids[dep_id] = ids

        # Workload IDs are unique by platform design too — always overwrite.
        if wl_id:
            cards[wl_id] = card
            key_types[wl_id] = "workload"
            registry_ids[wl_id] = ids

        # External IDs may have duplicates — apply the configured strategy.
        if ext_id:
            if ext_id not in cards:
                cards[ext_id] = card
                key_types[ext_id] = "external"
                registry_ids[ext_id] = ids
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
                    key_types[ext_id] = "external"
                    registry_ids[ext_id] = ids
                # "first" — keep existing entry (no-op)
    return ParsedRegistryCards(cards=cards, key_types=key_types, registry_ids=registry_ids)


class AgentCardRegistry:
    """Batch-capable, TTL-cached client for the central agent card registry.

    IDs are ``register()``-ed synchronously at config-parse time (no I/O).
    The first ``get()`` flushes all pending IDs in ≤3 HTTP calls — one per ID
    kind (deployment, external, workload), never mixed in a single request, plus
    one extra call per 20 IDs of the same kind (the API's per-parameter cap).
    Subsequent ``get()`` calls hit the in-memory cache until the soft TTL
    (``AGENT_CARD_REGISTRY_CACHE_TTL``) expires.  When a refresh fails, a cached
    card may still be returned while it remains within ``AGENT_CARD_REGISTRY_CACHE_TTL``.
    """

    def __init__(
        self,
        api_token: str | None = None,
        endpoint: str | None = None,
        timeout: float | None = None,
        cache_ttl: int | None = None,
        on_duplicate: DuplicateStrategy | None = None,
        cache_backend: AgentCardCacheBackend | None = None,
    ) -> None:
        self._api_token = api_token
        self._endpoint = endpoint
        self._lock = asyncio.Lock()

        # Pending registrations (filled synchronously, flushed on first get)
        self._pending_deployment_ids: set[str] = set()
        self._pending_external_ids: set[str] = set()
        self._pending_workload_ids: set[str] = set()

        # All IDs registered at config-parse time (used for background refresh)
        self._registered_deployment_ids: set[str] = set()
        self._registered_external_ids: set[str] = set()
        self._registered_workload_ids: set[str] = set()

        config = AgentCardRegistryConfig()
        self._timeout = timeout if timeout is not None else config.agent_card_registry_timeout
        self._cache_ttl = (
            cache_ttl if cache_ttl is not None else config.agent_card_registry_cache_ttl
        )
        self._on_duplicate: DuplicateStrategy = (
            on_duplicate if on_duplicate is not None else config.agent_card_registry_on_duplicate
        )
        self._backend = cache_backend or create_agent_card_cache_backend(config)

        logger.debug(
            "AgentCardRegistry created (cache_ttl=%ds, l2=%s)",
            self._cache_ttl,
            isinstance(self._backend, LayeredAgentCardCacheBackend),
        )

    # ------------------------------------------------------------------
    # Registration (synchronous — called at config-parse time)
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        deployment_id: str | None = None,
        external_id: str | None = None,
        workload_id: str | None = None,
    ) -> None:
        """Declare intent to look up an agent card.

        Call this at config-parse/validation time so that the first
        :meth:`get` can batch all pending IDs into a single prefetch.

        Exactly one of ``deployment_id``, ``external_id`` or ``workload_id``
        must be given.
        """
        if deployment_id:
            self._pending_deployment_ids.add(deployment_id)
            self._registered_deployment_ids.add(deployment_id)
        elif external_id:
            self._pending_external_ids.add(external_id)
            self._registered_external_ids.add(external_id)
        elif workload_id:
            self._pending_workload_ids.add(workload_id)
            self._registered_workload_ids.add(workload_id)

    def has_registered_lookups(self) -> bool:
        """Return whether any registry lookup IDs were registered."""
        return bool(
            self._registered_deployment_ids
            or self._registered_external_ids
            or self._registered_workload_ids
        )

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    async def _fetch(self, params: dict[str, str]) -> ParsedRegistryCards:
        """Fetch and parse all agent cards matching *params*.

        Splits ID lists longer than the API's per-parameter cap
        (:data:`_MAX_IDS_PER_REQUEST`) into several requests, accumulating the
        raw entries so that the ``on_duplicate`` strategy is applied once
        across the full result set — the same way pagination is accumulated
        inside :meth:`_fetch_entries`.
        """
        all_entries: list[dict[str, Any]] = []
        for chunk_params in _chunk_id_params(params):
            all_entries.extend(await self._fetch_entries(chunk_params))

        parsed = _parse_registry_response({"data": all_entries}, on_duplicate=self._on_duplicate)
        logger.info("Fetched %d agent card(s) from registry.", len(parsed.cards))
        return parsed

    async def _fetch_entries(self, params: dict[str, str]) -> list[dict[str, Any]]:
        """Execute a single registry HTTP GET with pagination, returning raw entries.

        Requests the maximum page size (100) to minimise round-trips, then
        follows ``next`` links until all pages are consumed.
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

        logger.debug(
            "Fetched %d registry entry(ies) over %d page(s).",
            len(all_entries),
            pages_fetched,
        )
        return all_entries

    async def _is_fresh(self, key: str, *, key_type: LookupKeyType) -> bool:
        """Return True if *key* is cached and within the soft TTL."""
        record = await self._backend.get_fresh(
            key,
            cache_ttl=self._cache_ttl,
            key_type=key_type,
        )
        return record is not None

    async def _try_get_stale(self, key: str, *, key_type: LookupKeyType) -> AgentCard | None:
        """Return a stale cached card when refresh failed."""
        record = await self._backend.get_stale(
            key,
            max_staleness_seconds=self._cache_ttl,
            key_type=key_type,
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
        await self._backend.store(
            parsed.cards,
            key_types=parsed.key_types,
            registry_ids=parsed.registry_ids,
        )

    def _age_cache_entry_for_test(self, lookup_key: str, seconds: float) -> None:
        """Shift a cached entry's fetch time backward (tests only)."""
        backend = self._backend
        if isinstance(backend, MemoryAgentCardCacheBackend):
            backend.age_entry_for_test(lookup_key, seconds)
            return

        if isinstance(backend, LayeredAgentCardCacheBackend):
            backend.memory.age_entry_for_test(lookup_key, seconds)

    async def _uncached(self, ids: Iterable[str], *, key_type: LookupKeyType) -> list[str]:
        """Return the *ids* of *key_type* that are not cached within the soft TTL."""
        return [i for i in ids if not await self._is_fresh(i, key_type=key_type)]

    async def _flush_pending(self) -> None:
        """Batch-fetch all registered-but-uncached IDs.  Must be called under ``_lock``."""
        missing: dict[LookupKeyType, list[str]] = {
            "deployment": await self._uncached(self._pending_deployment_ids, key_type="deployment"),
            "external": await self._uncached(self._pending_external_ids, key_type="external"),
            "workload": await self._uncached(self._pending_workload_ids, key_type="workload"),
        }

        # Clear pending sets regardless — they've been processed
        self._pending_deployment_ids.clear()
        self._pending_external_ids.clear()
        self._pending_workload_ids.clear()

        await self._fetch_and_store_by_kind(missing)

    async def _fetch_and_store_by_kind(
        self,
        ids_by_kind: dict[LookupKeyType, list[str]],
    ) -> None:
        """Fetch and cache each ID kind in its **own** request.

        Kinds are never combined: ``deploymentIds`` + ``workloadIds`` is an HTTP
        400, and either mixed with ``externalIds`` AND-matches to nothing.
        """
        for key_type, ids in ids_by_kind.items():
            if not ids:
                continue
            parsed = await self._fetch({_id_param_for(key_type): ",".join(ids)})
            await self._store_cards(parsed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def prefetch(
        self,
        *,
        deployment_ids: list[str] | None = None,
        external_ids: list[str] | None = None,
        workload_ids: list[str] | None = None,
    ) -> None:
        """Batch-fetch and cache agent cards in a minimum number of HTTP calls.

        Issues **separate** requests per ID kind: ``deploymentIds`` together
        with ``workloadIds`` is rejected with HTTP 400, and either combined with
        ``externalIds`` AND-matches and returns nothing.

        Already-cached (non-expired) IDs are skipped.

        Parameters
        ----------
        deployment_ids:
            Deployment IDs to prefetch.
        external_ids:
            External IDs to prefetch.
        workload_ids:
            Workload IDs to prefetch.
        """
        async with self._lock:
            missing: dict[LookupKeyType, list[str]] = {
                "deployment": await self._uncached(deployment_ids or [], key_type="deployment"),
                "external": await self._uncached(external_ids or [], key_type="external"),
                "workload": await self._uncached(workload_ids or [], key_type="workload"),
            }

            if not any(missing.values()):
                logger.debug("All requested agent cards already cached — skipping prefetch.")
                return

            await self._fetch_and_store_by_kind(missing)

    async def refresh_all_registered(self) -> None:
        """Re-fetch registered IDs whose cache entries are past the soft TTL.

        Failures are logged and existing cache entries are left in place so
        stale-if-error can continue serving them during registry outages.
        """
        if not self.has_registered_lookups():
            logger.debug("No registered agent card IDs; skipping background refresh.")
            return

        deployment_ids = sorted(self._registered_deployment_ids)
        external_ids = sorted(self._registered_external_ids)
        workload_ids = sorted(self._registered_workload_ids)
        logger.debug(
            "Refreshing registered agent cards "
            "(deployment_ids=%s, external_ids=%s, workload_ids=%s)",
            deployment_ids,
            external_ids,
            workload_ids,
        )
        try:
            await self.prefetch(
                deployment_ids=deployment_ids or None,
                external_ids=external_ids or None,
                workload_ids=workload_ids or None,
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
        workload_id: str | None = None,
    ) -> AgentCard:
        """Return a single agent card, using the cache or fetching on demand.

        On the first call, all IDs previously passed to :meth:`register`
        are batch-fetched in a single prefetch (one HTTP call per ID kind).
        Subsequent calls for already-cached, non-expired cards are instant.

        Exactly one of ``deployment_id``, ``external_id`` or ``workload_id``
        must be provided.

        Raises
        ------
        AgentCardRegistryError
            If the card cannot be found or the request fails.
        """
        provided = [i for i in (deployment_id, external_id, workload_id) if i]
        if len(provided) != 1:
            raise AgentCardRegistryError(
                "Specify exactly one of 'deployment_id', 'external_id' or 'workload_id'."
            )

        lookup_key: str = provided[0]
        lookup_key_type: LookupKeyType = (
            "deployment" if deployment_id else "external" if external_id else "workload"
        )

        # Fast path — fresh cache hit
        if fresh := await self._backend.get_fresh(
            lookup_key,
            cache_ttl=self._cache_ttl,
            key_type=lookup_key_type,
        ):
            return fresh.card

        async with self._lock:
            # Double-check after acquiring lock
            if fresh := await self._backend.get_fresh(
                lookup_key,
                cache_ttl=self._cache_ttl,
                key_type=lookup_key_type,
            ):
                return fresh.card

            try:
                # Flush all pending registrations in a batch
                if (
                    self._pending_deployment_ids
                    or self._pending_external_ids
                    or self._pending_workload_ids
                ):
                    await self._flush_pending()
                    if fresh := await self._backend.get_fresh(
                        lookup_key,
                        cache_ttl=self._cache_ttl,
                        key_type=lookup_key_type,
                    ):
                        return fresh.card

                # Still not fresh — fetch individually
                params: dict[str, str] = {_id_param_for(lookup_key_type): lookup_key}
                parsed = await self._fetch(params)
                await self._store_cards(parsed)

                if lookup_key in parsed.cards:
                    return parsed.cards[lookup_key]

                if fresh := await self._backend.get_fresh(
                    lookup_key,
                    cache_ttl=self._cache_ttl,
                    key_type=lookup_key_type,
                ):
                    return fresh.card

                # Successful miss — evict stale entry so stale-if-error cannot
                # resurrect a deregistered agent on a later fetch failure.
                await self._backend.evict(lookup_key, key_type=lookup_key_type)
            except AgentCardRegistryError:
                if stale_card := await self._try_get_stale(lookup_key, key_type=lookup_key_type):
                    return stale_card
                raise

        # Fetch succeeded but the requested key was absent from the response.
        raise AgentCardRegistryError(
            f"No agent card found in the central registry for "
            f"{lookup_key_type}_id='{lookup_key}'. Verify that the agent exists and is "
            "registered in your organisation."
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
