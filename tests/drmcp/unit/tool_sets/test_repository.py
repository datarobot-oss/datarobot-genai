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

"""Unit tests for the Tool Set repository."""

import pytest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from datarobot_genai.drmcp.core.tool_sets.tool_sets_repo import ToolSetRepository
from datarobot_genai.drmcp.core.tool_sets import ToolSet


@pytest.mark.asyncio
async def test_create_indexes_calls_collection_create_indexes() -> None:
    """GIVEN a repository, WHEN create_indexes runs, THEN indexes are created on the collection."""
    collection = MagicMock()
    collection.create_indexes = AsyncMock()
    db = MagicMock()
    db.__getitem__.return_value = collection

    await ToolSetRepository.new(db).create_indexes()

    db.__getitem__.assert_called_once_with(ToolSet.__collection__)
    collection.create_indexes.assert_awaited_once()
    index_names = [index.document["name"] for index in collection.create_indexes.await_args.args[0]]
    assert index_names == ["tool_sets_owner_name_unique", "tool_sets_by_owner"]
