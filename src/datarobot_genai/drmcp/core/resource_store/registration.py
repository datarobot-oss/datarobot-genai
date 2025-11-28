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

"""Registration helpers for ResourceStore with FastMCP."""

import logging
import tempfile
from typing import Any

from fastmcp import FastMCP

from .backends.filesystem import FilesystemBackend
from .resource_manager import ResourceStoreBackedResourceManager
from .store import ResourceStore

logger = logging.getLogger(__name__)

# Global ResourceManager instance (set during initialization)
_resource_manager: ResourceStoreBackedResourceManager | None = None


def get_resource_manager() -> ResourceStoreBackedResourceManager | None:
    """Get the global ResourceStoreBackedResourceManager instance."""
    return _resource_manager


def initialize_resource_store(
    mcp: FastMCP,
    backend: Any | None = None,
    storage_path: str | None = None,
    default_scope_id: str | None = None,
) -> ResourceStoreBackedResourceManager:
    """
    Initialize ResourceStore and register ResourceManager with FastMCP.

    This should be called during server startup (e.g., in pre_server_start lifecycle hook).

    Args:
        mcp: FastMCP instance
        backend: Optional ResourceBackend instance (if None, creates FilesystemBackend)
        storage_path: Path for FilesystemBackend (if backend is None)
        default_scope_id: Default scope ID for resources

    Returns
    -------
        ResourceStoreBackedResourceManager instance

    Example:
        ```python
        class MyLifecycle(BaseServerLifecycle):
            async def pre_server_start(self, mcp: FastMCP) -> None:
                initialize_resource_store(mcp, storage_path="/tmp/resources")
        ```
    """
    global _resource_manager  # noqa: PLW0603

    # Create backend if not provided
    if backend is None:
        if storage_path is None:
            storage_path = tempfile.mkdtemp(prefix="drmcp_resources_")
            logger.info(f"Using temporary storage path: {storage_path}")

        backend = FilesystemBackend(storage_path)

    # Create ResourceStore
    store = ResourceStore(backend)

    # Create ResourceStoreBackedResourceManager
    resource_manager = ResourceStoreBackedResourceManager(
        store=store,
        default_scope_id=default_scope_id,
    )

    # Store globally for access by tools
    _resource_manager = resource_manager

    logger.info("ResourceStore initialized and ResourceManager registered")

    # Note: FastMCP's mcp.add_resource() still works, but tools can now use
    # ResourceManager() which will use ResourceStore as backend.
    # To make tools use our ResourceManager, they should call:
    #   from datarobot_genai.drmcp.core.resource_store.registration import get_resource_manager
    #   resource_manager = get_resource_manager() or ResourceManager()
    #   resource_manager.add_resource(resource, data=data)

    return resource_manager

