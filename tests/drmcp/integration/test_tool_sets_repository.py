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

r"""Integration tests for ToolSetRepository against a real MongoDB instance.

Running locally
---------------
    docker run -d --name mongo-drmcp -p 27017:27017 mongo:7
    export MONGO_URI=mongodb://localhost:27017
    uv sync --extra drmcp --dev
    uv run pytest tests/drmcp/integration/test_tool_sets_repository.py -v
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from datetime import UTC
from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from urllib.parse import urlunparse

import pytest
from motor.motor_asyncio import AsyncIOMotorClient
from msf.dataclasses.fields import ObjectId
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

import datarobot_genai.drmcp.db_migrations.migration_20260722120000_create_tool_sets as migration
from datarobot_genai.drmcp.core.tool_sets import ToolEntry
from datarobot_genai.drmcp.core.tool_sets import ToolSet
from datarobot_genai.drmcp.core.tool_sets.tool_sets_repo import ToolSetRepository

_TEST_DB_NAME = "drmcp_migration_test"


def _host_mongo_uri(mongo_uri: str) -> str:
    """Strip any database path so tests can target a dedicated DB name."""
    parsed = urlparse(mongo_uri)
    return urlunparse((parsed.scheme, parsed.netloc, "", parsed.params, parsed.query, parsed.fragment))


@pytest.fixture
async def repository() -> AsyncGenerator[ToolSetRepository, None]:
    """Provide a repository wired to a disposable test database."""
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        pytest.skip("MONGO_URI not set — skipping tool_sets repository integration test")

    host_uri = _host_mongo_uri(mongo_uri)
    sync_client: MongoClient[Any] = MongoClient(host_uri)
    sync_client.drop_database(_TEST_DB_NAME)
    migration.run(sync_client[_TEST_DB_NAME])
    sync_client.close()

    motor_client = AsyncIOMotorClient(host_uri)
    db = motor_client[_TEST_DB_NAME]
    repo = ToolSetRepository.new(db)
    try:
        yield repo
    finally:
        motor_client.close()
        cleanup_client: MongoClient[Any] = MongoClient(host_uri)
        cleanup_client.drop_database(_TEST_DB_NAME)
        cleanup_client.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_indexes_is_idempotent(repository: ToolSetRepository) -> None:
    """GIVEN a migrated database, WHEN create_indexes runs twice, THEN no error is raised."""
    await repository.create_indexes()
    await repository.create_indexes()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_and_get_tool_set(repository: ToolSetRepository) -> None:
    """GIVEN a repository, WHEN a tool set is created, THEN it can be fetched by id."""
    owner_id = ObjectId()
    tool_set = ToolSet(
        name="integration-test-set",
        tools=[ToolEntry(name="catalog_list_datasets")],
        created_by=owner_id,
        created_at=datetime.now(UTC),
    )
    created = await repository.create(tool_set)
    assert created.id is not None

    fetched = await repository.get(created.id)
    assert fetched is not None
    assert fetched.name == "integration-test-set"
    assert fetched.created_by == owner_id
    assert [tool.name for tool in fetched.tools] == ["catalog_list_datasets"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_duplicate_owner_name_raises(repository: ToolSetRepository) -> None:
    """GIVEN an existing tool set, WHEN another with the same owner+name is created, THEN it fails."""
    owner_id = ObjectId()
    now = datetime.now(UTC)
    first = ToolSet(
        name="duplicate-name",
        tools=[ToolEntry(name="catalog_list_datasets")],
        created_by=owner_id,
        created_at=now,
    )
    duplicate = ToolSet(
        name="duplicate-name",
        tools=[ToolEntry(name="modeling_get_modeldetails")],
        created_by=owner_id,
        created_at=now,
    )
    await repository.create(first)
    with pytest.raises(DuplicateKeyError):
        await repository.create(duplicate)
