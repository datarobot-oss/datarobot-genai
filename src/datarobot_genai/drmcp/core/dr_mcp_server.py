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

import asyncio
import glob
import importlib
import logging
import os
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from starlette.middleware import Middleware

from .config import get_config
from .credentials import get_credentials
from .dynamic_tools.deployment.register import register_tools_of_datarobot_deployments
from .logging import MCPLogging
from .mcp_instance import mcp
from .memory_management import MemoryManager
from .routes import register_routes
from .routes_utils import prefix_mount_path
from .telemetry import OtelASGIMiddleware
from .telemetry import initialize_telemetry


def _import_modules_from_dir(directory: str, package_prefix: str) -> None:
    """Dynamically import all modules from a directory."""
    if not os.path.exists(directory):
        return
    for file in glob.glob(os.path.join(directory, "*.py")):
        if os.path.basename(file) != "__init__.py":
            module_name = f"{package_prefix}.{os.path.splitext(os.path.basename(file))[0]}"
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                logging.warning(f"Failed to import module {module_name}: {e}")


class BaseServerLifecycle:
    """
    Base server lifecycle interface with safe default implementations.

    This class provides hooks that are called at different stages of the server lifecycle.
    Subclasses can override any or all of these methods to add custom behavior.
    All methods have safe no-op defaults, so you only need to implement what you need.

    Lifecycle Order:
        1. pre_server_start()  - Before server initialization
        2. Server starts
        3. post_server_start() - After server is ready
        4. Server runs...
        5. Shutdown signal received
        6. pre_server_shutdown() - Before server cleanup
        7. Server stops

    Example:
        ```python
        class MyLifecycle(BaseServerLifecycle):
            async def pre_server_start(self, mcp: FastMCP) -> None:
                # Initialize resources
                self.db = await connect_database()

            async def pre_server_shutdown(self, mcp: FastMCP) -> None:
                # Clean up resources
                await self.db.close()

            # post_server_start not implemented - will use safe default (no-op)
        ```
    """

    async def pre_server_start(self, mcp: FastMCP) -> None:
        """
        Call before the server starts.

        Use this to:
        - Initialize resources
        - Set up connections
        - Validate configuration
        - Prepare server state

        Args:
            mcp: The FastMCP instance that will be started

        Note:
            Override this method in your subclass to add custom initialization.
            The default implementation is a safe no-op.
        """
        pass

    async def post_server_start(self, mcp: FastMCP) -> None:
        """
        Call after the server has started and is ready to handle requests.

        Use this to:
        - Register additional handlers
        - Start background tasks
        - Initialize delayed resources
        - Log startup completion

        Args:
            mcp: The running FastMCP instance

        Note:
            Override this method in your subclass to add post-startup logic.
            The default implementation is a safe no-op.
        """
        pass

    async def pre_server_shutdown(self, mcp: FastMCP) -> None:
        """
        Call before the server shuts down.

        Use this to:
        - Close database connections
        - Save application state
        - Clean up temporary files
        - Stop background tasks
        - Release resources

        Args:
            mcp: The running FastMCP instance

        Note:
            Override this method in your subclass to add cleanup logic.
            The default implementation is a safe no-op.
            This is ALWAYS called, even on Ctrl+C or errors.
        """
        pass


class DataRobotMCPServer:
    """
    DataRobot MCP server implementation using FastMCP framework.

    This server can be extended by providing custom configuration, credentials,
    and lifecycle handlers.
    """

    def __init__(
        self,
        mcp: FastMCP,
        transport: str = "streamable-http",
        config_factory: Callable[[], Any] | None = None,
        credentials_factory: Callable[[], Any] | None = None,
        lifecycle: BaseServerLifecycle | None = None,
        additional_module_paths: list[tuple[str, str]] | None = None,
    ):
        """
        Initialize the server.

        Args:
            mcp: FastMCP instance
            transport: Transport type ("streamable-http" or "stdio")
            config_factory: Optional factory function for user config
            credentials_factory: Optional factory function for user credentials
            lifecycle: Optional lifecycle handler (defaults to BaseServerLifecycle())
            additional_module_paths: Optional list of (directory, package_prefix) tuples for
                loading additional modules
        """
        # Initialize config and logging
        self._config = get_config()
        MCPLogging(self._config.app_log_level)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info(f"Config initialized: {self._config}")

        # Initialize credentials
        self._credentials = get_credentials()
        self._logger.info("Credentials initialized")

        self._user_config = config_factory() if config_factory else None
        self._logger.info(f"User config initialized: {self._user_config}")
        self._user_credentials = credentials_factory() if credentials_factory else None
        self._logger.info("User credentials initialized")

        # Initialize lifecycle
        self._lifecycle = lifecycle if lifecycle else BaseServerLifecycle()
        self._logger.info("Lifecycle initialized")

        self._mcp = mcp
        self._mcp_transport = transport

        # Initialize telemetry
        initialize_telemetry(mcp)

        # Initialize memory manager if AWS credentials are available
        self._memory_manager: MemoryManager | None = None
        if self._config.enable_memory_management:
            if self._credentials.has_aws_credentials():
                self._logger.info("Initializing memory manager")
                try:
                    self._memory_manager = MemoryManager.get_instance()
                except Exception as e:
                    self._logger.error(f"Error initializing memory manager: {e}")
                    self._logger.info("Skipping memory manager initialization")
                    self._memory_manager = None
            else:
                self._logger.info(
                    "No AWS credentials found, skipping memory manager initialization"
                )

        # Load base library modules
        base_dir = os.path.dirname(os.path.dirname(__file__))
        _import_modules_from_dir(os.path.join(base_dir, "tools", "core"), "tools.core")
        if self._config.enable_predictive_tools:
            _import_modules_from_dir(
                os.path.join(base_dir, "tools", "predictive"), "tools.predictive"
            )

        _import_modules_from_dir(os.path.join(base_dir, "tools"), "tools")
        _import_modules_from_dir(os.path.join(base_dir, "prompts"), "prompts")
        _import_modules_from_dir(os.path.join(base_dir, "resources"), "resources")

        # Load memory management tools if available
        if self._memory_manager:
            _import_modules_from_dir(
                os.path.join(base_dir, "tools", "core", "memory_management"),
                "tools.core.memory_management",
            )

        # Load additional modules if provided
        if additional_module_paths:
            for directory, package_prefix in additional_module_paths:
                self._logger.info(f"Loading additional modules from {directory}")
                _import_modules_from_dir(directory, package_prefix)

        # Register HTTP routes if using streamable-http transport
        if transport == "streamable-http":
            register_routes(self._mcp)

    def run(self, show_banner: bool = False) -> None:
        """Run the DataRobot MCP server synchronously."""
        try:
            # Validate configuration
            if not self._credentials.has_datarobot_credentials():
                self._logger.error("DataRobot credentials not configured")
                raise ValueError("Missing required DataRobot credentials")

            if self._config.mcp_server_register_dynamic_tools_on_startup:
                self._logger.info("Registering dynamic tools from deployments...")
                asyncio.run(register_tools_of_datarobot_deployments())

            # List registered tools, prompts, and resources before starting server
            tools = asyncio.run(self._mcp._mcp_list_tools())
            prompts = asyncio.run(self._mcp._mcp_list_prompts())
            resources = asyncio.run(self._mcp._mcp_list_resources())

            self._logger.info(f"Registered tools: {len(tools)}")
            for tool in tools:
                self._logger.info(f" > {tool.name}")
            self._logger.info(f"Registered prompts: {len(prompts)}")
            for prompt in prompts:
                self._logger.info(f" > {prompt.name}")
            self._logger.info(f"Registered resources: {len(resources)}")
            for resource in resources:
                self._logger.info(f" > {resource.name}")

            # Execute pre-server start actions
            asyncio.run(self._lifecycle.pre_server_start(self._mcp))

            # Create event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def run_server(show_banner: bool = show_banner) -> None:
                # Start server in background based on transport type
                if self._mcp_transport == "stdio":
                    server_task = asyncio.create_task(
                        self._mcp.run_stdio_async(show_banner=show_banner)
                    )
                elif self._mcp_transport == "streamable-http":
                    server_task = asyncio.create_task(
                        self._mcp.run_http_async(
                            transport="http",
                            middleware=[Middleware(OtelASGIMiddleware)],
                            show_banner=show_banner,
                            port=self._config.mcp_server_port,
                            log_level=self._config.mcp_server_log_level,
                            host=self._config.mcp_server_host,
                            stateless_http=True,
                            path=prefix_mount_path("/mcp"),
                        )
                    )
                else:
                    raise ValueError(f"Unsupported transport: {self._mcp_transport}")

                # Give the server a moment to initialize
                await asyncio.sleep(1)

                # Execute post-server start actions
                await self._lifecycle.post_server_start(self._mcp)

                # Wait for server to complete
                await server_task

            # Start the server
            self._logger.info("Starting MCP server...")
            try:
                loop.run_until_complete(run_server(show_banner=show_banner))
            except KeyboardInterrupt:
                self._logger.info("Server interrupted by user")
            finally:
                # Execute pre-shutdown actions
                self._logger.info("Shutting down server...")
                loop.run_until_complete(self._lifecycle.pre_server_shutdown(self._mcp))
                loop.close()

        except Exception as e:
            self._logger.error(f"Server error: {e}")
            raise


def create_mcp_server(
    config_factory: Callable[[], Any] | None = None,
    credentials_factory: Callable[[], Any] | None = None,
    lifecycle: BaseServerLifecycle | None = None,
    additional_module_paths: list[tuple[str, str]] | None = None,
    transport: str = "streamable-http",
) -> DataRobotMCPServer:
    """
    Create a DataRobot MCP server.

    Args:
        config_factory: Optional factory function for user config
        credentials_factory: Optional factory function for user credentials
        lifecycle: Optional lifecycle handler
        additional_module_paths: Optional list of (directory, package_prefix) tuples
        transport: Transport type ("streamable-http" or "stdio")

    Returns
    -------
        Configured DataRobotMCPServer instance

    Example:
        ```python
        # Basic usage with defaults
        server = create_mcp_server()
        server.run()

        # With custom configuration
        from myapp.config import get_my_config
        from myapp.lifecycle import MyLifecycle

        server = create_mcp_server(
            config_factory=get_my_config,
            lifecycle=MyLifecycle(),
            additional_module_paths=[
                ("/path/to/my/tools", "myapp.tools"),
                ("/path/to/my/prompts", "myapp.prompts"),
            ]
        )
        server.run()
        ```
    """
    # Use the global mcp instance that tools are registered with

    return DataRobotMCPServer(
        mcp=mcp,
        transport=transport,
        config_factory=config_factory,
        credentials_factory=credentials_factory,
        lifecycle=lifecycle,
        additional_module_paths=additional_module_paths,
    )
