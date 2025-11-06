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

from fastmcp import FastMCP


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
