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
from typing import Any
from typing import ParamSpec
from typing import TypeVar

P = ParamSpec("P")
R = TypeVar("R")

# Global registry to store tool metadata
_TOOL_REGISTRY: list[tuple[Callable, dict[str, Any]]] = []


def tool_metadata(**metadata: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Store tool metadata to a function without MCP registration.

    Records the function and its metadata in a registry that drmcp discovers via
    :func:`get_registered_tools` to register the tool. The function is returned
    unchanged — no wrapper — so the registry and direct callers see the real
    function (including its true coroutine-ness).

    Args:
        **metadata: Keyword arguments for tool metadata (tags, name, description, etc.)

    Returns
    -------
        Decorator that registers the function and attaches its metadata, then
        returns it unchanged.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        _TOOL_REGISTRY.append((func, metadata))
        # Expose the metadata on the function for direct access; the function
        # itself is returned untouched so callers and FastMCP see the original.
        func._tool_metadata = metadata  # type: ignore[attr-defined]
        return func

    return decorator


def get_registered_tools() -> list[tuple[Callable, dict[str, Any]]]:
    """Get all registered tools and their metadata.

    Returns
    -------
        List of (function, metadata) tuples.
    """
    return _TOOL_REGISTRY.copy()
