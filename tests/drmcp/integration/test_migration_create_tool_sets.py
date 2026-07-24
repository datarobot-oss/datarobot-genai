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

r"""Integration tests for the create_tool_sets migration.

These tests require a real MongoDB instance.  They are marked
``@pytest.mark.integration`` and are excluded from the default ``task test``
and ``task drmcp-unit`` runs (which ignore ``tests/drmcp/integration``).

Running locally
---------------
    docker run -d --name mongo-drmcp -p 27017:27017 mongo:7
    export MONGO_URI=mongodb://localhost:27017/drmcp_migration_test
    uv sync --extra drmcp --dev
    uv run pytest tests/drmcp/integration/test_migration_create_tool_sets.py -v
"""

import os
from collections.abc import Generator
from typing import Any

import pytest
from msf.dataclasses.fields import ObjectId
from pymongo import MongoClient
from pymongo.database import Database

import datarobot_genai.drmcp.db_migrations.migration_20260722120000_create_tool_sets as migration
from datarobot_genai.drmcp.core.tool_sets import ToolSet

_TEST_DB_NAME = "drmcp_migration_test"
_COLLECTION = ToolSet.__collection__


@pytest.fixture
def db() -> Generator[Database[Any], None, None]:
    """Connect to a real MongoDB, drop the test DB before and after each test."""
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        pytest.skip("MONGO_URI not set — skipping migration integration test")

    client: MongoClient[Any] = MongoClient(mongo_uri)
    database = client[_TEST_DB_NAME]
    client.drop_database(_TEST_DB_NAME)
    yield database
    client.drop_database(_TEST_DB_NAME)
    client.close()


@pytest.mark.integration
def test_run_creates_collection(db: Database[Any]) -> None:
    """GIVEN an empty database, WHEN the migration runs, THEN the collection exists."""
    migration.run(db)
    assert _COLLECTION in db.list_collection_names()


@pytest.mark.integration
def test_run_creates_unique_owner_name_index(db: Database[Any]) -> None:
    """GIVEN a fresh DB, WHEN the migration runs, THEN the owner+name unique index is created."""
    migration.run(db)
    index_names = {idx["name"] for idx in db[_COLLECTION].list_indexes()}
    assert "tool_sets_owner_name_unique" in index_names


@pytest.mark.integration
def test_run_creates_owner_index(db: Database[Any]) -> None:
    """GIVEN a fresh DB, WHEN the migration runs, THEN the created_by index exists."""
    migration.run(db)
    index_names = {idx["name"] for idx in db[_COLLECTION].list_indexes()}
    assert "tool_sets_by_owner" in index_names


@pytest.mark.integration
def test_unique_index_enforced(db: Database[Any]) -> None:
    """GIVEN the migration has run, WHEN two docs with the same (created_by, name) are inserted.

    THEN the second insert raises a DuplicateKeyError.
    """
    from pymongo.errors import DuplicateKeyError

    migration.run(db)
    owner_id = ObjectId()
    doc = {
        "created_by": owner_id,
        "name": "my-set",
        "tools": [{"name": "catalog_list_datasets"}],
        "created_at": "2026-07-22T12:00:00Z",
    }
    db[_COLLECTION].insert_one(doc.copy())
    with pytest.raises(DuplicateKeyError):
        db[_COLLECTION].insert_one(doc.copy())


@pytest.mark.integration
def test_run_is_idempotent(db: Database[Any]) -> None:
    """GIVEN the migration has already run, WHEN it runs again, THEN no error is raised."""
    migration.run(db)
    migration.run(db)
    assert _COLLECTION in db.list_collection_names()


@pytest.mark.integration
def test_rollback_drops_collection(db: Database[Any]) -> None:
    """GIVEN the migration has run, WHEN rollback runs, THEN the collection is gone."""
    migration.run(db)
    assert _COLLECTION in db.list_collection_names()
    migration.rollback(db)
    assert _COLLECTION not in db.list_collection_names()


@pytest.mark.integration
def test_rollback_on_empty_db_is_safe(db: Database[Any]) -> None:
    """GIVEN no collection exists, WHEN rollback runs, THEN no error is raised."""
    migration.rollback(db)


@pytest.mark.integration
def test_analyze_returns_migration_analysis(db: Database[Any]) -> None:
    """GIVEN a fresh DB, WHEN analyze runs, THEN a MigrationAnalysis is returned."""
    result = migration.analyze(db)
    assert result is not None
    assert len(result.collections_impacted) == 1
    assert result.collections_impacted[0].collection_name == _COLLECTION
    assert result.collections_impacted[0].total_documents == 0
