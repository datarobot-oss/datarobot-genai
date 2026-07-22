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

"""Unit tests for Tool Set domain models.

No MongoDB connection required — all tests operate on plain Python objects.
"""

from datetime import UTC
from datetime import datetime

import pytest
from msf.dataclasses.fields import ObjectId
from pydantic import ValidationError

from datarobot_genai.drmcp.core.tool_sets import ToolEntry
from datarobot_genai.drmcp.core.tool_sets import ToolSet


def test_collection_name_on_entity() -> None:
    """GIVEN ToolSet, WHEN __collection__ is read, THEN it matches the agreed name."""
    assert ToolSet.__collection__ == "tool_sets"


def test_tool_entry_defaults_version_to_none() -> None:
    """GIVEN a ToolEntry with only a name, WHEN created, THEN version is None."""
    entry = ToolEntry(name="catalog_list_datasets")
    assert entry.name == "catalog_list_datasets"
    assert entry.version is None


def test_tool_set_validates_full_schema() -> None:
    """GIVEN valid field values, WHEN ToolSet is constructed, THEN all schema fields are set."""
    tool_set_id = ObjectId()
    user_id = ObjectId()
    created_at = datetime(2026, 7, 6, 12, 0, 0, tzinfo=UTC)
    tool_set = ToolSet(
        id=tool_set_id,
        name="data-science-workbench",
        description="Core modeling and catalog tools for exploratory workflows",
        tools=[
            ToolEntry(name="modeling_get_modeldetails", version="1.0.0"),
            ToolEntry(name="catalog_list_datasets"),
        ],
        created_by=user_id,
        created_at=created_at,
    )

    assert tool_set.id == tool_set_id
    assert tool_set.name == "data-science-workbench"
    assert tool_set.description == "Core modeling and catalog tools for exploratory workflows"
    assert tool_set.created_by == user_id
    assert tool_set.created_at == created_at
    assert len(tool_set.tools) == 2
    assert tool_set.tools[0].name == "catalog_list_datasets"
    assert tool_set.tools[0].version is None
    assert tool_set.tools[1].name == "modeling_get_modeldetails"
    assert tool_set.tools[1].version == "1.0.0"


def test_tool_set_rejects_empty_tools() -> None:
    """GIVEN an empty tools list, WHEN ToolSet is created, THEN validation fails."""
    with pytest.raises(ValidationError):
        ToolSet(
            name="empty-set",
            tools=[],
            created_by=ObjectId(),
            created_at=datetime(2026, 7, 22, 12, 0, 0, tzinfo=UTC),
        )


def test_tool_set_dedupes_and_sorts_tools() -> None:
    """GIVEN duplicate tool names, WHEN ToolSet is created, THEN tools are unique and sorted."""
    now = datetime(2026, 7, 6, 12, 0, 0, tzinfo=UTC)
    tool_set = ToolSet(
        name="deployment-ops",
        tools=[
            ToolEntry(name="deployment_list_deployments"),
            ToolEntry(name="modeling_get_modeldetails"),
            ToolEntry(name="modeling_get_modeldetails", version="2.0.0"),
        ],
        created_by=ObjectId(),
        created_at=now,
    )
    assert [tool.name for tool in tool_set.tools] == [
        "deployment_list_deployments",
        "modeling_get_modeldetails",
    ]
