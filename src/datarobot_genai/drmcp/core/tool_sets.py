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

"""Tool set persistence, CRUD service, and resolution helpers.

Tech spec §3 and §4.  No caching — every call goes to MongoDB.

Data shapes
-----------
ToolSet          — Mongo persistence model (internal).
ToolSetSummary   — Lean REST response returned by POST (create) and GET (list).
ToolSetDetail    — Full REST response returned by GET /{toolSetId}/;
                   extends ToolSetSummary with resolved GalleryTool objects.
PaginatedResult  — Typed wrapper for paginated list responses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import TYPE_CHECKING
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo.errors import DuplicateKeyError

from datarobot_genai.drmcp.core.tool_gallery import GalleryTool
from datarobot_genai.drmcp.core.tool_gallery import build_tool
from datarobot_genai.drmcp.core.tool_gallery import is_tool_visible_to_caller

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persistence model
# ---------------------------------------------------------------------------


@dataclass
class ToolSet:
    """Mongo document shape for a tool set."""

    id: ObjectId | None
    name: str
    created_by: str          # user ID string (from auth_ctx.user.id)
    tool_names: list[str]    # sorted, deduplicated
    created_at: datetime


# ---------------------------------------------------------------------------
# REST response models
# ---------------------------------------------------------------------------


@dataclass
class ToolSetSummary:
    """Lean response — returned by POST create and list rows."""

    id: str
    name: str
    created_by: str
    created_at: int   # Unix-ms
    tool_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "createdBy": self.created_by,
            "createdAt": self.created_at,
            "toolCount": self.tool_count,
        }


@dataclass
class ToolSetDetail(ToolSetSummary):
    """Full response — returned by GET /{toolSetId}/."""

    tools: list[GalleryTool]
    unresolved_tool_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["tools"] = [t.to_dict() for t in self.tools]
        base["unresolvedToolNames"] = self.unresolved_tool_names
        return base


@dataclass
class PaginatedResult:
    items: list[ToolSet]
    total_count: int
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total_count


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _doc_to_tool_set(doc: dict) -> ToolSet:
    return ToolSet(
        id=doc["_id"],
        name=doc["name"],
        created_by=doc["created_by"],
        tool_names=doc.get("tool_names", []),
        created_at=doc["created_at"],
    )


def _tool_set_to_summary(ts: ToolSet) -> ToolSetSummary:
    ts_id = str(ts.id) if ts.id else ""
    created_at_ms = int(ts.created_at.timestamp() * 1000)
    return ToolSetSummary(
        id=ts_id,
        name=ts.name,
        created_by=ts.created_by,
        created_at=created_at_ms,
        tool_count=len(ts.tool_names),
    )


# ---------------------------------------------------------------------------
# CRUD service
# ---------------------------------------------------------------------------


class ToolSetNotFoundError(Exception):
    pass


class ToolSetNameConflictError(Exception):
    pass


class ToolSetCRUDService:
    """Async motor-backed CRUD for tool sets.  No in-memory caching."""

    COLLECTION = "tool_sets"
    DEFAULT_LIMIT = 100

    def __init__(self, db: "AsyncIOMotorDatabase") -> None:  # type: ignore[type-arg]
        self._col = db[self.COLLECTION]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def insert(self, tool_set: ToolSet) -> ToolSet:
        """Insert a new tool set.

        Raises :class:`ToolSetNameConflictError` if a set with the same
        ``(created_by, name)`` already exists.
        """
        doc = {
            "name": tool_set.name,
            "created_by": tool_set.created_by,
            "tool_names": tool_set.tool_names,
            "created_at": tool_set.created_at,
        }
        try:
            result = await self._col.insert_one(doc)
        except DuplicateKeyError:
            raise ToolSetNameConflictError(
                f"A tool set named '{tool_set.name}' already exists for this user."
            )
        tool_set.id = result.inserted_id
        return tool_set

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def find_by_id(
        self, *, created_by: str, tool_set_id: str
    ) -> ToolSet | None:
        """Return the tool set owned by *created_by* with Mongo id *tool_set_id*."""
        try:
            oid = ObjectId(tool_set_id)
        except InvalidId:
            return None
        doc = await self._col.find_one({"_id": oid, "created_by": created_by})
        return _doc_to_tool_set(doc) if doc else None

    async def find_by_names(
        self, *, created_by: str, names: list[str]
    ) -> list[ToolSet]:
        """Return all tool sets owned by *created_by* whose name is in *names*."""
        cursor = self._col.find({"created_by": created_by, "name": {"$in": names}})
        return [_doc_to_tool_set(doc) async for doc in cursor]

    async def find_paginated(
        self,
        *,
        created_by: str,
        name: str | None = None,
        search: str | None = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> PaginatedResult:
        """Return a paginated list of tool sets owned by *created_by*."""
        query: dict[str, Any] = {"created_by": created_by}
        if name is not None:
            query["name"] = name
        elif search is not None:
            query["name"] = {"$regex": search, "$options": "i"}

        total_count = await self._col.count_documents(query)
        cursor = self._col.find(query).skip(offset).limit(limit)
        items = [_doc_to_tool_set(doc) async for doc in cursor]

        return PaginatedResult(
            items=items,
            total_count=total_count,
            limit=limit,
            offset=offset,
        )


# ---------------------------------------------------------------------------
# Resolution helper (shared by POST validation and GET /{toolSetId}/)
# ---------------------------------------------------------------------------


async def resolve_tool_set_tools(
    mcp: Any,
    tool_set: ToolSet,
    *,
    caller_created_by: str | None,
) -> tuple[list[GalleryTool], list[str]]:
    """Resolve stored tool names to enriched GalleryTool objects.

    Returns ``(resolved_tools, unresolved_tool_names)``.
    Tools not in the registry or not visible to the caller appear in
    *unresolved_tool_names* (not in *resolved_tools*).
    """
    registry = {t.name: t for t in await mcp.list_tools()}
    resolved: list[GalleryTool] = []
    unresolved: list[str] = []

    for tool_name in sorted(tool_set.tool_names):
        mcp_tool = registry.get(tool_name)
        if mcp_tool is None or not is_tool_visible_to_caller(
            mcp_tool, caller_created_by=caller_created_by
        ):
            unresolved.append(tool_name)
        else:
            resolved.append(build_tool(mcp_tool, caller_created_by=caller_created_by))

    return resolved, unresolved


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_tool_set(
    *,
    name: str,
    created_by: str,
    tool_names: list[str],
) -> ToolSet:
    """Create a new unsaved :class:`ToolSet` from validated inputs."""
    deduped = sorted(set(tool_names))
    return ToolSet(
        id=None,
        name=name,
        created_by=created_by,
        tool_names=deduped,
        created_at=datetime.now(tz=timezone.utc),
    )
