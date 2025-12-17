# Copyright 2025 DataRobot, Inc.
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

"""Filesystem backend implementation for ResourceStore."""

import json
import logging
import sqlite3
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any

from ..backend import ResourceBackend
from ..models import Resource
from ..models import Scope

logger = logging.getLogger(__name__)


class BackendError(Exception):
    """Base exception for backend errors."""

    pass


class FilesystemBackend(ResourceBackend):
    """
    Filesystem-based backend using SQLite for metadata and files for content.

    Storage structure:
        {base_path}/
            resources.db          # SQLite database for metadata
            content/              # Directory for resource content
                {resource_id}    # Files named by resource ID
    """

    def __init__(self, base_path: str | Path) -> None:
        """
        Initialize filesystem backend.

        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.content_dir = self.base_path / "content"
        self.content_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.base_path / "resources.db"
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resources (
                    id TEXT PRIMARY KEY,
                    scope_type TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    lifetime TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    ttl_seconds INTEGER,
                    content_type TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    content_ref TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scope ON resources(scope_type, scope_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_kind ON resources(kind)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_lifetime ON resources(lifetime)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at ON resources(created_at)
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _resource_to_row(self, resource: Resource) -> tuple:
        """Convert Resource to database row."""
        return (
            resource.id,
            resource.scope.type,
            resource.scope.id,
            resource.kind,
            resource.lifetime,
            resource.createdAt.isoformat(),
            resource.ttlSeconds,
            resource.contentType,
            json.dumps(resource.metadata, default=str),
            resource.contentRef,
        )

    def _row_to_resource(self, row: tuple) -> Resource:
        """Convert database row to Resource."""
        (
            id_val,
            scope_type,
            scope_id,
            kind,
            lifetime,
            created_at_str,
            ttl_seconds,
            content_type,
            metadata_json,
            content_ref,
        ) = row

        return Resource(
            id=id_val,
            scope=Scope(type=scope_type, id=scope_id),
            kind=kind,
            lifetime=lifetime,
            createdAt=datetime.fromisoformat(created_at_str),
            ttlSeconds=ttl_seconds,
            contentType=content_type,
            metadata=json.loads(metadata_json),
            contentRef=content_ref,
        )

    async def put(self, resource: Resource, data: bytes | str | None) -> Resource:
        """Store a resource and its data."""
        try:
            # Write content to file if provided
            content_path = self.content_dir / resource.id
            if data is not None:
                if isinstance(data, str):
                    content_path.write_text(data, encoding="utf-8")
                else:
                    content_path.write_bytes(data)
                # Update contentRef to point to the file
                resource.contentRef = str(content_path)
            elif not resource.contentRef:
                # If no data, ensure contentRef is set (might be a reference to external storage)
                resource.contentRef = str(content_path)

            # Store metadata in database
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO resources
                    (id, scope_type, scope_id, kind, lifetime, created_at, ttl_seconds,
                     content_type, metadata, content_ref)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._resource_to_row(resource),
                )
                conn.commit()
            finally:
                conn.close()

            logger.debug(f"Stored resource {resource.id} at {content_path}")
            return resource

        except Exception as e:
            logger.error(f"Failed to store resource {resource.id}: {e}")
            raise BackendError(f"Failed to store resource: {e}") from e

    async def get(self, resource_id: str) -> tuple[Resource, bytes | str | None] | None:
        """Retrieve a resource and its data."""
        try:
            # Get metadata from database
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    """
                    SELECT id, scope_type, scope_id, kind, lifetime, created_at, ttl_seconds,
                           content_type, metadata, content_ref
                    FROM resources
                    WHERE id = ?
                    """,
                    (resource_id,),
                )
                row = cursor.fetchone()
            finally:
                conn.close()

            if not row:
                return None

            resource = self._row_to_resource(row)

            # Load content from file if it exists
            content_path = Path(resource.contentRef)
            data: bytes | str | None = None
            if content_path.exists():
                # Determine if content should be text or binary based on content type
                is_text = (
                    resource.contentType.startswith("text/")
                    or resource.contentType == "application/json"
                )
                if is_text:
                    data = content_path.read_text(encoding="utf-8")
                else:
                    data = content_path.read_bytes()

            return (resource, data)

        except Exception as e:
            logger.error(f"Failed to get resource {resource_id}: {e}")
            raise BackendError(f"Failed to get resource: {e}") from e

    async def query(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[Resource]:
        """Query resources by filters."""
        try:
            filters = filters or {}
            conn = sqlite3.connect(self.db_path)
            try:
                conditions = []
                params: list[Any] = []

                # Filter by scope
                if "scope" in filters:
                    scope = filters["scope"]
                    if isinstance(scope, dict):
                        scope_type = scope.get("type")
                        scope_id = scope.get("id")
                    elif isinstance(scope, Scope):
                        scope_type = scope.type
                        scope_id = scope.id
                    else:
                        raise ValueError("scope filter must be a dict or Scope object")

                    if scope_type:
                        conditions.append("scope_type = ?")
                        params.append(scope_type)
                    if scope_id:
                        conditions.append("scope_id = ?")
                        params.append(scope_id)

                # Filter by kind
                if "kind" in filters:
                    conditions.append("kind = ?")
                    params.append(filters["kind"])

                # Filter by lifetime
                if "lifetime" in filters:
                    conditions.append("lifetime = ?")
                    params.append(filters["lifetime"])

                # Filter by metadata (simple key-value matching)
                if "metadata" in filters and isinstance(filters["metadata"], dict):
                    # For filesystem backend, we'll do post-filtering for metadata
                    # since SQLite JSON support varies
                    pass

                where_clause = " AND ".join(conditions) if conditions else "1=1"
                query = f"""
                    SELECT id, scope_type, scope_id, kind, lifetime, created_at, ttl_seconds,
                           content_type, metadata, content_ref
                    FROM resources
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                """

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                resources = [self._row_to_resource(row) for row in rows]

                # Post-filter by metadata if needed
                if "metadata" in filters and isinstance(filters["metadata"], dict):
                    metadata_filters = filters["metadata"]
                    resources = [
                        r
                        for r in resources
                        if all(r.metadata.get(k) == v for k, v in metadata_filters.items())
                    ]

                return resources

            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Failed to query resources: {e}")
            raise BackendError(f"Failed to query resources: {e}") from e

    async def delete(self, resource_id: str) -> None:
        """Delete a resource and its data."""
        try:
            # Get resource to find content path
            result = await self.get(resource_id)
            if not result:
                return

            resource, _ = result

            # Delete content file
            content_path = Path(resource.contentRef)
            if content_path.exists():
                content_path.unlink()

            # Delete metadata from database
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("DELETE FROM resources WHERE id = ?", (resource_id,))
                conn.commit()
            finally:
                conn.close()

            logger.debug(f"Deleted resource {resource_id}")

        except Exception as e:
            logger.error(f"Failed to delete resource {resource_id}: {e}")
            raise BackendError(f"Failed to delete resource: {e}") from e

    async def cleanup_expired(self) -> int:
        """Clean up expired ephemeral resources."""
        try:
            now = datetime.now(timezone.utc)
            conn = sqlite3.connect(self.db_path)
            try:
                # Find expired ephemeral resources
                cursor = conn.execute(
                    """
                    SELECT id, created_at, ttl_seconds
                    FROM resources
                    WHERE lifetime = 'ephemeral' AND ttl_seconds IS NOT NULL
                    """
                )
                rows = cursor.fetchall()

                expired_ids = []
                for row in rows:
                    resource_id, created_at_str, ttl_seconds = row
                    created_at = datetime.fromisoformat(created_at_str)
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    expires_at = created_at + timedelta(seconds=ttl_seconds or 0)

                    if now >= expires_at:
                        expired_ids.append(resource_id)

                # Delete expired resources
                count = 0
                for resource_id in expired_ids:
                    await self.delete(resource_id)
                    count += 1

                return count

            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Failed to cleanup expired resources: {e}")
            raise BackendError(f"Failed to cleanup expired resources: {e}") from e
