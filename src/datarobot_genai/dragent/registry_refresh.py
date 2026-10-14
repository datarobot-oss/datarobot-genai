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

"""Background refresh loop for the central agent card registry cache."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from typing import Protocol

from datarobot_genai.dragent.agent_card_registry import AgentCardRegistry
from datarobot_genai.dragent.agent_card_registry import get_default_registry
from datarobot_genai.dragent.registry_warmup import collect_registry_lookup_ids
from datarobot_genai.dragent.registry_warmup import register_registry_lookup_ids

if TYPE_CHECKING:
    from nat.data_models.config import Config

logger = logging.getLogger(__name__)

_MIN_REFRESH_INTERVAL_SECONDS = 60

# Upper bound for exponential backoff after consecutive refresh failures.
_MAX_FAILURE_BACKOFF_SECONDS = 15 * 60

# Jitter multiplier range applied to every sleep (healthy or backoff).
_JITTER_MIN = 0.5
_JITTER_MAX = 1.5

# Cap the exponent so ``base * 2**failures`` cannot overflow before ``min(cap, …)``.
_MAX_BACKOFF_EXPONENT = 16


class _SupportsUniform(Protocol):
    def uniform(self, a: float, b: float) -> float:
        """Return a random float in ``[a, b]``."""


def background_refresh_interval(soft_cache_ttl: int) -> int:
    """Return the background refresh poll interval for *soft_cache_ttl*.

    Polls at half the soft TTL (minimum 60s) so expired entries are picked up
    soon after they go stale rather than waiting up to another full soft TTL.
    """
    return max(_MIN_REFRESH_INTERVAL_SECONDS, soft_cache_ttl // 2)


def refresh_sleep_seconds(
    base_interval: int,
    soft_cache_ttl: int,
    failures: int,
    *,
    rng: _SupportsUniform | None = None,
) -> float:
    """Return the next background-refresh sleep duration.

    Healthy polls (*failures* == 0) sleep ``base_interval`` with ±50% jitter.
    After consecutive failures the delay grows as ``base * 2**failures``,
    capped at ``min(15m, max(base, soft_ttl))``, then jittered the same way.
    """
    rng = rng or random.Random()
    if failures <= 0:
        delay = float(base_interval)
    else:
        cap = min(_MAX_FAILURE_BACKOFF_SECONDS, max(base_interval, soft_cache_ttl))
        exponent = min(failures, _MAX_BACKOFF_EXPONENT)
        delay = min(cap, float(base_interval) * (2**exponent))
    return delay * rng.uniform(_JITTER_MIN, _JITTER_MAX)


async def registry_refresh_loop(
    registry: AgentCardRegistry,
    interval_seconds: int,
) -> None:
    """Periodically refresh soft-expired registered agent cards.

    Sleeps a jittered interval before each attempt. Consecutive failures
    exponentially increase the next delay (still jittered) so many agents do
    not keep hammering Control Hub in lockstep during an outage.
    """
    failures = 0
    while True:
        delay = refresh_sleep_seconds(
            interval_seconds,
            registry.soft_cache_ttl,
            failures,
        )
        await asyncio.sleep(delay)
        try:
            await registry.refresh_all_registered()
        except Exception:
            failures += 1
            logger.warning(
                "Background agent card registry refresh failed "
                "(consecutive_failures=%d); keeping cached entries and backing off",
                failures,
                exc_info=True,
            )
        else:
            failures = 0


@asynccontextmanager
async def registry_refresh_lifespan(config: Config) -> AsyncIterator[None]:
    """Start the background refresh task for the registry singleton.

    No-op when no registry-backed remote A2A clients are configured.
    """
    collected = collect_registry_lookup_ids(config)
    if collected.is_empty():
        logger.debug("No registry-backed A2A function groups; skipping background refresh task.")
        yield
        return

    registry = await get_default_registry()
    register_registry_lookup_ids(registry, collected)

    if registry.soft_cache_ttl == 0:
        logger.debug(
            "Agent card registry caching disabled (soft_cache_ttl=0); "
            "skipping background refresh task."
        )
        yield
        return

    refresh_interval = background_refresh_interval(registry.soft_cache_ttl)
    logger.info(
        "Starting agent card registry background refresh (interval=%ds, soft_cache_ttl=%ds)",
        refresh_interval,
        registry.soft_cache_ttl,
    )
    task = asyncio.create_task(registry_refresh_loop(registry, refresh_interval))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        logger.debug("Agent card registry background refresh task stopped.")
