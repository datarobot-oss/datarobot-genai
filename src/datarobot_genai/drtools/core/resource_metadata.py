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

from collections.abc import Callable
from typing import Any
from typing import ParamSpec
from typing import TypeVar

P = ParamSpec("P")
R = TypeVar("R")

# Global registry to store resource metadata
_RESOURCE_REGISTRY: list[tuple[Callable, dict[str, Any]]] = []


def resource_metadata(**metadata: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Store MCP-resource metadata on a function without MCP registration.

    Records the function and its metadata in a registry that each MCP server
    discovers via :func:`get_registered_resources` to register the resource onto
    its own FastMCP instance. The function is returned unchanged — no wrapper —
    so the server registers the real handler and FastMCP sees its true
    coroutine-ness.

    Args:
        **metadata: Keyword arguments for resource metadata. ``uri`` is required
            by the MCP resource protocol; ``name``, ``title``, ``description``,
            ``mime_type`` and ``tags`` are optional.

    Returns
    -------
        Decorator that registers the function and attaches its metadata, then
        returns it unchanged.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        _RESOURCE_REGISTRY.append((func, metadata))
        # Expose the metadata on the function for direct access; the function
        # itself is returned untouched so callers and FastMCP see the original.
        func._resource_metadata = metadata  # type: ignore[attr-defined]
        return func

    return decorator


def get_registered_resources() -> list[tuple[Callable, dict[str, Any]]]:
    """Get all registered resources and their metadata.

    Returns
    -------
        List of (function, metadata) tuples.
    """
    return _RESOURCE_REGISTRY.copy()
