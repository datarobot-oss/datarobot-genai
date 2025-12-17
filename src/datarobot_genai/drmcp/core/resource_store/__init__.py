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

"""Unified ResourceStore system for conversation state, memory, and MCP resources."""

import logging
import tempfile

from fastmcp import FastMCP

from .backend import ResourceBackend
from .backends.filesystem import FilesystemBackend
from .models import Lifetime
from .models import Resource
from .models import Scope
from .models import ScopeType
from .resource_manager import ResourceStoreBackedResourceManager
from .store import ResourceStore

logger = logging.getLogger(__name__)


def initialize_resource_store(mcp: FastMCP, storage_path: str | None = None) -> None:
    """
    Initialize ResourceStore and replace FastMCP's ResourceManager.

    Args:
        mcp: FastMCP instance
        storage_path: Path for resource storage. If None, uses a temporary directory.
    """
    if storage_path is None:
        storage_path = tempfile.mkdtemp(prefix="drmcp_resources_")
        logger.info(f"Using temporary storage path for ResourceStore: {storage_path}")
    else:
        logger.info(f"Using configured storage path for ResourceStore: {storage_path}")

    backend = FilesystemBackend(storage_path)
    store = ResourceStore(backend)
    resource_manager = ResourceStoreBackedResourceManager(
        store=store,
        default_scope_id=None,  # Can be set per-conversation
    )

    # Replace FastMCP's ResourceManager with our ResourceStore-backed one
    mcp._resource_manager = resource_manager
    logger.info("Replaced FastMCP's _resource_manager with ResourceStoreBackedResourceManager")


__all__ = [
    "ResourceStore",
    "ResourceBackend",
    "Resource",
    "Scope",
    "ScopeType",
    "Lifetime",
    "ResourceStoreBackedResourceManager",
    "initialize_resource_store",
]
