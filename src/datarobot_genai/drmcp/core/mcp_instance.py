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

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any
from typing import overload

from fastmcp import Context
from fastmcp import FastMCP
from fastmcp.exceptions import NotFoundError
from fastmcp.tools import FunctionTool
from fastmcp.tools import Tool
from fastmcp.utilities.types import NotSet
from fastmcp.utilities.types import NotSetT
from mcp.types import AnyFunction
from mcp.types import Tool as MCPTool
from mcp.types import ToolAnnotations

from .config import MCPServerConfig
from .config import get_config
from .logging import log_execution
from .memory_management import MemoryManager
from .memory_management import get_memory_manager
from .telemetry import trace_execution
from .tool_filter import filter_tools_by_tags
from .tool_filter import list_all_tags

logger = logging.getLogger(__name__)


async def get_agent_and_storage_ids(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[str | None, str | None]:
    """
    Extract agent ID from request context and get corresponding storage ID.

    Args:
        args: Positional arguments that may contain a Context object
        kwargs: Keyword arguments that may contain a Context object

    Returns
    -------
        Tuple of (agent_id, storage_id), both may be None if not found
    """
    # Find the context argument if it exists
    ctx = next((arg for arg in args if isinstance(arg, Context)), kwargs.get("ctx"))

    # Extract X-Agent-Id if context and headers exist
    agent_id = None
    if (
        ctx
        and ctx.request_context
        and ctx.request_context.request
        and hasattr(ctx.request_context.request, "headers")
    ):
        headers = ctx.request_context.request.headers
        agent_id = headers.get("x-agent-id")

    # If agent_id was found, get the active storage_id
    storage_id = None
    if agent_id and MemoryManager.is_initialized():
        memory_manager = get_memory_manager()
        if memory_manager:
            storage_id = await memory_manager.get_active_storage_id_for_agent(agent_id)

    return agent_id, storage_id


class TaggedFastMCP(FastMCP):
    """Extended FastMCP that supports tags, deployments and other annotations directly in the
    tool decorator.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._deployments_map: dict[str, str] = {}

    @overload
    def tool(
        self,
        name_or_fn: AnyFunction,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: set[str] | None = None,
        output_schema: dict[str, Any] | None | NotSetT = NotSet,
        annotations: ToolAnnotations | dict[str, Any] | None = None,
        exclude_args: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        enabled: bool | None = None,
    ) -> FunctionTool: ...

    @overload
    def tool(
        self,
        name_or_fn: str | None = None,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: set[str] | None = None,
        output_schema: dict[str, Any] | None | NotSetT = NotSet,
        annotations: ToolAnnotations | dict[str, Any] | None = None,
        exclude_args: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        enabled: bool | None = None,
    ) -> Callable[[AnyFunction], FunctionTool]: ...

    def tool(
        self,
        name_or_fn: str | Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: set[str] | None = None,
        output_schema: dict[str, Any] | None | NotSetT = NotSet,
        annotations: ToolAnnotations | dict[str, Any] | None = None,
        exclude_args: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        enabled: bool | None = None,
        **kwargs: Any,
    ) -> Callable[[AnyFunction], FunctionTool] | FunctionTool:
        """
        Extend tool decorator that supports tags and other annotations, while remaining
        signature-compatible with FastMCP.tool to avoid recursion issues with partials.
        """
        if isinstance(annotations, dict):
            annotations = ToolAnnotations(**annotations)

        # Ensure tags are available both via native fastmcp `tags` and inside annotations
        if tags is not None:
            tags_ = sorted(tags)
            if annotations is None:
                annotations = ToolAnnotations()  # type: ignore[call-arg]
                annotations.tags = tags_  # type: ignore[attr-defined, union-attr]
            else:
                # At this point, annotations is ToolAnnotations (not dict)
                assert isinstance(annotations, ToolAnnotations)
                annotations.tags = tags_  # type: ignore[attr-defined]

        return super().tool(
            name_or_fn,
            name=name,
            title=title,
            description=description,
            tags=tags,
            output_schema=output_schema
            if output_schema is not None
            else kwargs.get("output_schema"),
            annotations=annotations,
            exclude_args=exclude_args,
            meta=meta,
            enabled=enabled,
        )

    async def list_tools(
        self, tags: list[str] | None = None, match_all: bool = False
    ) -> list[MCPTool]:
        """
        List all available tools, optionally filtered by tags.

        Args:
            tags: Optional list of tags to filter by. If None, returns all tools.
            match_all: If True, tool must have all specified tags (AND logic).
                      If False, tool must have at least one tag (OR logic).
                      Only used when tags is provided.

        Returns
        -------
            List of MCPTool objects that match the tag criteria.
        """
        # Get all tools from the parent class
        all_tools = await super()._mcp_list_tools()

        # If no tags specified, return all tools
        if not tags:
            return all_tools

        # Filter tools by tags
        filtered_tools = filter_tools_by_tags(list(all_tools), tags, match_all)

        return filtered_tools  # type: ignore[return-value]

    async def get_all_tags(self) -> list[str]:
        """
        Get all unique tags from all registered tools.

        Returns
        -------
            List of all unique tags sorted alphabetically.
        """
        all_tools = await self._mcp_list_tools()
        return list_all_tags(list(all_tools))

    async def get_deployment_mapping(self) -> dict[str, str]:
        """
        Get the list of deployment IDs for all registered dynamic tools.

        Returns
        -------
            Dictionary mapping deployment IDs to tool names.
        """
        return self._deployments_map.copy()

    async def set_deployment_mapping(self, deployment_id: str, tool_name: str) -> None:
        """
        Add or update the mapping of a deployment ID to a tool name.

        Args:
            deployment_id: The ID of the deployment.
            tool_name: The name of the tool associated with the deployment.
        """
        existing = self._deployments_map.get(deployment_id)
        if existing and existing != tool_name:
            logger.debug(
                f"Deployment ID {deployment_id} already mapped to {existing}, updating to "
                f"{tool_name}"
            )
            try:
                self.remove_tool(existing)
            except NotFoundError:
                logger.debug(f"Tool {existing} not found in registry, skipping removal")
        self._deployments_map[deployment_id] = tool_name

    async def remove_deployment_mapping(self, deployment_id: str) -> None:
        """
        Remove the mapping of a deployment ID to a tool name.

        Args:
            deployment_id: The ID of the deployment to remove.
        """
        removed = self._deployments_map.pop(deployment_id, None)
        if removed is not None:
            logger.debug(f"Removed deployment mapping for ID {deployment_id} with tool {removed}")
            try:
                self.remove_tool(removed)
            except NotFoundError:
                logger.debug(f"Tool {removed} not found in registry, skipping removal")


# Create the tagged MCP instance
mcp_server_configs: MCPServerConfig = get_config()

mcp = TaggedFastMCP(
    name=mcp_server_configs.mcp_server_name,
    on_duplicate_tools=mcp_server_configs.tool_registration_duplicate_behavior,
)


def dr_core_mcp_tool(
    name: str | None = None,
    description: str | None = None,
    tags: set[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Combine decorator that includes mcp.tool() and dr_mcp_extras()."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        instrumented = dr_mcp_extras()(func)
        mcp.tool(name=name, description=description, tags=tags)(instrumented)
        return instrumented

    return decorator


async def memory_aware_wrapper(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Add memory management capabilities to any async function.
    Extracts agent and storage IDs from the context and adds them to kwargs if found.

    Args:
        func: The async function to wrap
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns
    -------
        The result of calling the wrapped function
    """
    # Get agent and storage IDs from context
    agent_id, storage_id = await get_agent_and_storage_ids(args, kwargs)

    # Add IDs to kwargs if found
    if agent_id and storage_id:
        kwargs["agent_id"] = agent_id
        kwargs["storage_id"] = storage_id

    # Call the original function
    return await func(*args, **kwargs)


def dr_mcp_tool(
    name: str | None = None,
    description: str | None = None,
    tags: set[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Combine decorator that includes mcp.tool(), dr_mcp_extras(), and capture memory ids from
    the request headers if they exist.

    Args:
        name: Tool name
        description: Tool description
        tags: Optional set of tags to apply to the tool
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await memory_aware_wrapper(func, *args, **kwargs)

        # Apply the MCP decorators
        instrumented = dr_mcp_extras()(wrapper)
        mcp.tool(name=name, description=description, tags=tags)(instrumented)
        return instrumented

    return decorator


def dr_mcp_extras(
    type: str = "tool",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Combine decorator that includes log_execution and trace_execution().

    Args:
        type: default is "tool", other options are "prompt", "resource"
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return log_execution(trace_execution(trace_type=type)(func))

    return decorator


async def register_tools(
    fn: AnyFunction,
    name: str | None = None,
    title: str | None = None,
    description: str | None = None,
    tags: set[str] | None = None,
    deployment_id: str | None = None,
) -> Tool:
    """
    Register new tools after server has started.

    Args:
        fn: The function to register as a tool
        name: Optional name for the tool (defaults to function name)
        title: Optional human-readable title for the tool
        description: Optional description of what the tool does
        tags: Optional set of tags to apply to the tool
        deployment_id: Optional deployment ID associated with the tool

    Returns
    -------
        The registered Tool object
    """
    tool_name = name or fn.__name__
    logger.info(f"Registering new tool: {tool_name}")

    # Create a memory-aware version of the function
    @wraps(fn)
    async def memory_aware_fn(*args: Any, **kwargs: Any) -> Any:
        return await memory_aware_wrapper(fn, *args, **kwargs)

    # Apply dr_mcp_extras to the memory-aware function
    wrapped_fn = dr_mcp_extras()(memory_aware_fn)

    # Create annotations with tags, deployment_id if provided
    annotations = ToolAnnotations()  # type: ignore[call-arg]
    if tags is not None:
        annotations.tags = tags  # type: ignore[attr-defined]
    if deployment_id is not None:
        annotations.deployment_id = deployment_id  # type: ignore[attr-defined]

    tool = Tool.from_function(
        fn=wrapped_fn,
        name=tool_name,
        title=title,
        description=description,
        annotations=annotations,
        tags=tags,
    )

    # Register the tool
    registered_tool = mcp.add_tool(tool)

    # Map deployment ID to tool name if provided
    if deployment_id:
        await mcp.set_deployment_mapping(deployment_id, tool_name)

    # Verify tool is registered
    tools = await mcp.list_tools()
    if not any(tool.name == tool_name for tool in tools):
        raise RuntimeError(f"Tool {tool_name} was not registered successfully")
    logger.info(f"Registered tools: {len(tools)}")

    return registered_tool
