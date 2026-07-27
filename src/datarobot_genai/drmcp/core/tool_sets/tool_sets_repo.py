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

"""MongoDB repository for Tool Sets."""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorDatabase
from msf.databases.mongodb.repository import MongoRepository
from msf.types import DictStrAny
from pymongo import ASCENDING
from pymongo import IndexModel

from datarobot_genai.drmcp.core.tool_sets.models import ToolSet

logger = logging.getLogger(__name__)


def tool_set_index_models() -> list[IndexModel]:
    """Index definitions for the ``tool_sets`` collection."""
    return [
        IndexModel(
            [("created_by", ASCENDING), ("name", ASCENDING)],
            unique=True,
            name="tool_sets_owner_name_unique",
        ),
        IndexModel(
            [("created_by", ASCENDING)],
            name="tool_sets_by_owner",
        ),
    ]


class ToolSetRepository(MongoRepository[ToolSet]):
    """Data access layer for Tool Sets persisted in MongoDB."""

    @classmethod
    def new(cls, db: AsyncIOMotorDatabase[DictStrAny]) -> ToolSetRepository:
        return cls(ToolSet, db)

    async def create_indexes(self) -> None:
        """Create indexes on the tool sets collection (idempotent)."""
        await self._collection.create_indexes(tool_set_index_models())  # noqa: MSF001
        logger.info("Indexes ensured on collection '%s'", ToolSet.__collection__)
