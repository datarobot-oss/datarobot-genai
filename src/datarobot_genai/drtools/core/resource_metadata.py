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

"""Resource metadata decorator for drtools.

The mirror image of :mod:`datarobot_genai.drtools.core.tool_metadata`, but for
MCP *resources*. It records metadata about a resource handler without creating
any dependency on ``drmcp`` or ``fastmcp`` (drtools must stay free of both).
Each MCP server (``drmcp``, ``global-mcp``) discovers these via
:func:`get_registered_resources` and wires them onto its own FastMCP instance
with its own resource decorator.

A resource's metadata must include the ``uri`` it is served at (required by the
MCP resource protocol); ``name``, ``title``, ``description``, ``mime_type`` and
``tags`` are optional.
"""

import inspect
from collections.abc import Awaitable
from collections.abc import Callable
from functools import wraps
from typing import Any
from typing import ParamSpec
from typing import TypeVar
from typing import cast

P = ParamSpec("P")
R = TypeVar("R")

# Global registry to store resource metadata
_RESOURCE_REGISTRY: list[tuple[Callable, dict[str, Any]]] = []


def resource_metadata(**metadata: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Store MCP-resource metadata on a function without MCP registration.

    This decorator stores the function and its metadata in a registry that can be
    discovered by an MCP server when it needs to register the resources.

    Args:
        **metadata: Keyword arguments for resource metadata. ``uri`` is required
            by the MCP resource protocol; ``name``, ``title``, ``description``,
            ``mime_type`` and ``tags`` are optional.

    Returns
    -------
        Decorator function that preserves the original function while storing metadata.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Store the function and metadata in the registry
        _RESOURCE_REGISTRY.append((func, metadata))

        if inspect.iscoroutinefunction(func):
            async_func = cast(Callable[P, Awaitable[R]], func)

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                return await async_func(*args, **kwargs)

            # Store metadata as an attribute for easy access
            async_wrapper._resource_metadata = metadata  # type: ignore[attr-defined]
            return cast(Callable[P, R], async_wrapper)

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        # Store metadata as an attribute for easy access
        sync_wrapper._resource_metadata = metadata  # type: ignore[attr-defined]

        return sync_wrapper

    return decorator


def get_registered_resources() -> list[tuple[Callable, dict[str, Any]]]:
    """Get all registered resources and their metadata.

    Returns
    -------
        List of (function, metadata) tuples.
    """
    return _RESOURCE_REGISTRY.copy()
