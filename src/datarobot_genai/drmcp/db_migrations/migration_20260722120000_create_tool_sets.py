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

"""Migration: create_tool_sets.

Created: 2026-07-22 12:00:00

Background/Rationale
====================
Tool Sets are user-owned, named collections of MCP tool references persisted
in a dedicated MongoDB database owned by the MCP server.

What the migration does
-----------------------
Creates the ``tool_sets`` collection and its two indexes:

- ``tool_sets_owner_name_unique`` — unique compound index on
  ``(created_by, name)``, enforcing that a user cannot create two tool sets
  with the same name.
- ``tool_sets_by_owner`` — single-field index on ``created_by`` for
  efficient per-user list queries.

Rollback
--------
Drops the entire ``tool_sets`` collection.

Expected runtime and performance considerations
-----------------------------------------------
Collection and index creation on an empty database is near-instant.
"""

import logging
from typing import Any

from dr_mongo_migrations.analysis import CollectionAnalysis
from dr_mongo_migrations.analysis import MigrationAnalysis
from pymongo.database import Database

from datarobot_genai.drmcp.core.tool_sets import ToolSet
from datarobot_genai.drmcp.core.tool_sets.tool_sets_repo import tool_set_index_models

logger = logging.getLogger(__name__)


def analyze(db: Database[Any]) -> MigrationAnalysis:
    """Report the current state of the ``tool_sets`` collection."""
    analysis = MigrationAnalysis(migration_name=__name__, collections_impacted=[])
    analysis.collections_impacted.append(
        CollectionAnalysis(
            db_name=db.name,
            collection_name=ToolSet.__collection__,
            documents_affected=0,
            total_documents=db[ToolSet.__collection__].estimated_document_count()
            if ToolSet.__collection__ in db.list_collection_names()
            else 0,
        )
    )
    return analysis


def run(db: Database[Any]) -> bool:
    """Create the ``tool_sets`` collection and its indexes."""
    logger.info("Applying migration: create %s", ToolSet.__collection__)
    db[ToolSet.__collection__].create_indexes(tool_set_index_models())
    logger.info("Collection '%s' and indexes created", ToolSet.__collection__)
    return True


def rollback(db: Database[Any]) -> bool:
    """Drop the ``tool_sets`` collection."""
    logger.info("Rolling back migration: dropping %s", ToolSet.__collection__)
    db.drop_collection(ToolSet.__collection__)
    logger.info("Collection '%s' dropped", ToolSet.__collection__)
    return True
