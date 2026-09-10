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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from datarobot_genai.dragent.agent_card_registry import AgentCardRegistry
from datarobot_genai.dragent.agent_card_registry import get_default_registry
from datarobot_genai.dragent.registry_warmup import collect_registry_lookup_ids

if TYPE_CHECKING:
    from nat.data_models.config import Config

logger = logging.getLogger(__name__)

# Refresh registered cards that are past the soft TTL this often.
_REFRESH_INTERVAL_SECONDS = 30 * 60


async def registry_refresh_loop(
    registry: AgentCardRegistry,
    interval_seconds: int = _REFRESH_INTERVAL_SECONDS,
) -> None:
    """Periodically refresh soft-expired registered agent cards."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await registry.refresh_all_registered()
        except Exception:
            logger.exception("Background agent card registry refresh failed")


@asynccontextmanager
async def registry_refresh_lifespan(config: Config) -> AsyncIterator[None]:
    """Start the background refresh task for the registry singleton.

    No-op when no registry-backed remote A2A clients are configured.
    """
    if collect_registry_lookup_ids(config).is_empty():
        logger.debug("No registry-backed A2A function groups; skipping background refresh task.")
        yield
        return

    registry = await get_default_registry()
    if not registry.has_registered_lookups():
        logger.debug("No registered agent card IDs; skipping background refresh task.")
        yield
        return

    logger.info(
        "Starting agent card registry background refresh (interval=%ds)",
        _REFRESH_INTERVAL_SECONDS,
    )
    task = asyncio.create_task(registry_refresh_loop(registry, _REFRESH_INTERVAL_SECONDS))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        logger.debug("Agent card registry background refresh task stopped.")
