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

"""Tool metadata decorator for drtools.

This module provides a decorator that stores metadata about tools without
creating any dependency on drmcp. The metadata can be discovered and used
by drmcp to register the tools.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any
from typing import TypeVar

T = TypeVar("T")

# Global registry to store tool metadata
_TOOL_REGISTRY: list[tuple[Callable, dict[str, Any]]] = []


def tool_metadata(**metadata: Any) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Store tool metadata to a function without MCP registration.

    This decorator stores the function and its metadata in a registry that can be
    discovered by drmcp when it needs to register the tools.

    Args:
        **metadata: Keyword arguments for tool metadata (tags, name, description, etc.)

    Returns
    -------
        Decorator function that preserves the original function while storing metadata.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Store the function and metadata in the registry
        _TOOL_REGISTRY.append((func, metadata))

        # Return the original function unchanged
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)

        # Store metadata as an attribute for easy access
        wrapper._tool_metadata = metadata  # type: ignore

        return wrapper

    return decorator


def get_registered_tools() -> list[tuple[Callable, dict[str, Any]]]:
    """Get all registered tools and their metadata.

    Returns
    -------
        List of (function, metadata) tuples.
    """
    return _TOOL_REGISTRY.copy()
