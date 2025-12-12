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

"""Backend interface for ResourceStore implementations."""

from abc import ABC
from abc import abstractmethod
from typing import Any

from .models import Resource


class ResourceBackend(ABC):
    """Abstract backend interface for resource storage."""

    @abstractmethod
    async def put(self, resource: Resource, data: bytes | str | None) -> Resource:
        """
        Store a resource and its data.

        Args:
            resource: Resource metadata
            data: Resource content (bytes, string, or None for metadata-only)

        Returns
        -------
            The stored resource (may have updated fields like contentRef)

        Raises
        ------
            BackendError: If storage operation fails
        """
        pass

    @abstractmethod
    async def get(self, resource_id: str) -> tuple[Resource, bytes | str | None] | None:
        """
        Retrieve a resource and its data.

        Args:
            resource_id: Unique resource identifier

        Returns
        -------
            Tuple of (resource, data) if found, None otherwise

        Raises
        ------
            BackendError: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def query(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[Resource]:
        """
        Query resources by filters.

        Args:
            filters: Dictionary with optional keys:
                - scope: Scope object or dict with 'type' and 'id'
                - kind: Resource kind string
                - lifetime: Lifetime string ('ephemeral' or 'persistent')
                - metadata: Dict of metadata key-value pairs to match

        Returns
        -------
            List of matching resources

        Raises
        ------
            BackendError: If query operation fails
        """
        pass

    @abstractmethod
    async def delete(self, resource_id: str) -> None:
        """
        Delete a resource and its data.

        Args:
            resource_id: Unique resource identifier

        Raises
        ------
            BackendError: If deletion operation fails
        """
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """
        Clean up expired ephemeral resources.

        Returns
        -------
            Number of resources cleaned up
        """
        pass

