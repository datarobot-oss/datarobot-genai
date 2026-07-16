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

"""Async MongoDB client for the MCP server.

Reads connection details from environment variables (same convention as the
DataRobot notebooks service):

    MONGO_URI   — MongoDB connection string  (required)
    DB_NAME     — Database name              (default: drmcp)

The module-level client is created lazily on the first call to ``get_db()``.
``ensure_indexes()`` is called at server startup as a safety net; production
deploys rely on the ``sbin/drmcp-migrate-db`` pre-install job instead.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorClient
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None  # type: ignore[type-arg]

_REPO_ROOT = Path(__file__).resolve().parents[6]
_INDEX_CONFIG = _REPO_ROOT / "config" / "mongo_indexes.json"


def _get_client() -> AsyncIOMotorClient:  # type: ignore[type-arg]
    global _client
    if _client is None:
        uri = os.environ.get("MONGO_URI")
        if not uri:
            raise RuntimeError(
                "MONGO_URI environment variable is not set. "
                "Set it to your MongoDB connection string (e.g. mongodb://localhost:27017)."
            )
        _client = AsyncIOMotorClient(uri)
        logger.info("MongoDB client created for URI %s", uri.split("@")[-1])
    return _client


async def get_db() -> AsyncIOMotorDatabase:  # type: ignore[type-arg]
    """Return the application database handle."""
    db_name = os.environ.get("DB_NAME", "drmcp")
    return _get_client()[db_name]


async def close_client() -> None:
    """Close the motor client.  Call from server shutdown hook."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB client closed.")


async def ensure_indexes() -> None:
    """Create indexes declared in config/mongo_indexes.json if absent.

    This is a startup safety net for development and POC environments.
    Production deploys run ``sbin/drmcp-migrate-db`` as a pre-install job
    which applies indexes before the server starts.
    """
    if not _INDEX_CONFIG.exists():
        logger.debug("Index config not found at %s — skipping ensure_indexes.", _INDEX_CONFIG)
        return

    with _INDEX_CONFIG.open() as fh:
        config: dict = json.load(fh)

    db = await get_db()
    for collection_name, indexes in config.items():
        collection = db[collection_name]
        try:
            existing = {idx["name"] async for idx in collection.list_indexes()}
        except Exception:
            existing = set()

        for idx_def in indexes:
            idx_name = idx_def["name"]
            if idx_name in existing:
                continue
            key = list(idx_def["key"].items())
            opts = {k: v for k, v in idx_def.items() if k != "key"}
            await collection.create_index(key, **opts)
            logger.info("Created index %s on collection %s.", idx_name, collection_name)
