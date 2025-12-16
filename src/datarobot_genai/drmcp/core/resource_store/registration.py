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

"""Registration helpers for ResourceStore with FastMCP.

ResourceStore is automatically initialized in DataRobotMCPServer.__init__.
This module provides utilities for accessing the ResourceManager from tools.
"""

from fastmcp import FastMCP

from datarobot_genai.drmcp.core.mcp_instance import mcp as global_mcp

from .resource_manager import ResourceStoreBackedResourceManager


def get_resource_manager(mcp: FastMCP | None = None) -> ResourceStoreBackedResourceManager | None:
    """
    Get the ResourceStoreBackedResourceManager instance.

    Args:
        mcp: Optional FastMCP instance. If provided, gets ResourceManager from it.
             If None, tries to get from the global mcp instance.

    Returns
    -------
        ResourceStoreBackedResourceManager instance or None if not found
    """
    if mcp is not None:
        return getattr(mcp, "_resource_manager", None)

    # Try to get from global mcp instance
    return getattr(global_mcp, "_resource_manager", None)
