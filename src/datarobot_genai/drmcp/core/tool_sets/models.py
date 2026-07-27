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

"""Domain models for Tool Sets persisted in MongoDB.

Data model
----------
ToolSet
    A named, user-owned collection of tool references.
ToolEntry
    A single tool reference: ``{name, version}``.
"""

from __future__ import annotations

from datetime import datetime

from msf.databases.mongodb.entity import MongoEntity
from msf.dataclasses import DataClass
from msf.dataclasses.fields import ObjectId
from pydantic import Field
from pydantic import field_validator


class ToolEntry(DataClass):
    """A single tool reference within a Tool Set."""

    name: str
    version: str | None = None


class ToolSet(MongoEntity):
    """A user-owned, named collection of tool references persisted in MongoDB."""

    __collection__ = "tool_sets"

    name: str
    description: str | None = None
    tools: list[ToolEntry] = Field(min_length=1)
    created_by: ObjectId
    created_at: datetime

    @field_validator("tools")
    @classmethod
    def tools_are_unique_and_sorted_by_name(cls, tools: list[ToolEntry]) -> list[ToolEntry]:
        """Deduplicate by tool name and sort alphabetically."""
        tools_by_name: dict[str, ToolEntry] = {}
        for tool in sorted(tools, key=lambda entry: entry.name):
            tools_by_name.setdefault(tool.name, tool)
        return list(tools_by_name.values())
