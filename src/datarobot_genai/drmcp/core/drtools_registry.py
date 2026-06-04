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

"""Registry loader for drtools functions decorated with tool_metadata / resource_metadata."""

import importlib
import logging
from collections.abc import Callable
from typing import Any

from datarobot_genai.drtools.core import get_registered_resources
from datarobot_genai.drtools.core import get_registered_tools

from .enums import DataRobotMCPResourceCategory
from .enums import DataRobotMCPToolCategory
from .mcp_instance import ResourceInitArguments
from .mcp_instance import dr_mcp_resource
from .mcp_instance import dr_mcp_tool

logger = logging.getLogger(__name__)


def register_drtools_function(func: Callable, metadata: dict[str, Any]) -> None:
    """Register a drtools tool function with the MCP server.

    Args:
        func: The function to register
        metadata: Tool metadata (tags, name, description, etc.)
    """
    # Apply the dr_mcp_tool decorator with the metadata
    dr_mcp_tool(tool_category=DataRobotMCPToolCategory.BUILT_IN_TOOL, **metadata)(func)

    logger.debug(f"Registered drtools tool: {func.__name__}")


def register_drtools_resource(func: Callable, metadata: dict[str, Any]) -> None:
    """Register a drtools resource function with the MCP server.

    Args:
        func: The resource handler to register
        metadata: Resource metadata (uri, name, description, mime_type, tags, ...)
    """
    resource_init_args = ResourceInitArguments(**metadata)
    dr_mcp_resource(
        resource_init_args,
        resource_category=DataRobotMCPResourceCategory.BUILT_IN_RESOURCE,
    )(func)

    logger.debug(f"Registered drtools resource: {func.__name__}")


def load_drtools_registry(module_name: str) -> None:
    """Load and register all tools and resources from a drtools module.

    Args:
        module_name: Full module name (e.g., 'datarobot_genai.drtools.panels.tools')
    """
    try:
        # Import the module first to trigger the @tool_metadata / @resource_metadata decorators
        if module_name.startswith("datarobot_genai.drtools.core."):
            logger.debug(f"Skipping module: {module_name}")
            return

        logger.debug(f"Importing module: {module_name}")
        importlib.import_module(module_name)

        # Register each tool defined in this module
        tool_count = 0
        for func, metadata in get_registered_tools():
            if func.__module__ == module_name:
                register_drtools_function(func, metadata)
                tool_count += 1

        # Register each resource defined in this module
        resource_count = 0
        for func, metadata in get_registered_resources():
            if func.__module__ == module_name:
                register_drtools_resource(func, metadata)
                resource_count += 1

        logger.debug(
            f"Registered {tool_count} tools and {resource_count} resources from {module_name}"
        )

    except ImportError as e:
        logger.debug(f"Could not import module {module_name}: {e}")
    except Exception as e:
        logger.error(f"Error loading drtools registry from {module_name}: {e}")
