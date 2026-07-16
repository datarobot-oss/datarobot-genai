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

"""
Migration: create tool_sets collection and its unique index.

v1 has no existing data to migrate — tool_sets is a brand-new collection.
This migration establishes the collection and the (created_by, name) unique
index so the migration ledger records the baseline schema version.

Compatible with the dr-mongo-migrations interface:
  analyze(db) -> dict   — dry-run description
  run(db)     -> bool   — apply (idempotent)
  rollback(db)-> bool   — revert
"""

COLLECTION = "tool_sets"
INDEX_NAME = "tool_sets_created_by_name_unique"


def analyze(db: object) -> dict:
    """Return a description of what this migration will do (dry-run)."""
    return {
        "migration": "20260706000000_create_tool_sets",
        "collections_affected": [COLLECTION],
        "description": (
            f"Create collection '{COLLECTION}' and unique compound index "
            f"({INDEX_NAME}) on (created_by, name)."
        ),
        "estimated_documents": 0,
    }


async def run(db: object) -> bool:  # type: ignore[override]
    """Apply the migration.

    Creates the tool_sets collection if it doesn't exist and ensures the
    unique (created_by, name) compound index is present.  Safe to run
    multiple times (idempotent).
    """
    collection = db[COLLECTION]

    existing = {idx["name"] async for idx in collection.list_indexes()}
    if INDEX_NAME not in existing:
        await collection.create_index(
            [("created_by", 1), ("name", 1)],
            unique=True,
            name=INDEX_NAME,
        )

    return True


async def rollback(db: object) -> bool:  # type: ignore[override]
    """Revert the migration.

    Drops the unique index.  The collection itself is left in place to
    avoid accidental data loss if documents were inserted between run and
    rollback.
    """
    collection = db[COLLECTION]
    try:
        await collection.drop_index(INDEX_NAME)
    except Exception:
        pass
    return True
