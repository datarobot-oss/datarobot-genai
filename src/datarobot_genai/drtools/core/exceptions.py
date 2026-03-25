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

"""Custom exceptions for DataRobot tools."""


class ToolError(Exception):
    """Drop-in replacement for fastmcp.exceptions.ToolError.

    This exception provides full compatibility with fastmcp's ToolError while
    removing the dependency on the fastmcp package. It should be raised when
    a tool encounters an error during execution that prevents it from completing
    its task.

    The MCP framework expects tools to raise ToolError for user-facing errors
    that should be reported back to the client. This is distinct from other
    exceptions which might indicate programming errors or system failures.

    Attributes
    ----------
        message: The error message to be displayed to the user.
        args: Tuple containing the error message (for compatibility with base Exception).

    Examples
    --------
        >>> raise ToolError("Failed to create deployment: No prediction servers available")
        >>> raise ToolError(f"Job failed with status {job.status}")
        >>> try:
        ...     risky_operation()
        ... except ValueError as e:
        ...     raise ToolError(f"Invalid input: {e}")
    """

    def __init__(self, message: str) -> None:
        """Initialize the ToolError with a descriptive message.

        Args:
            message: A clear, user-friendly description of what went wrong.
                    This message will be displayed to the end user.
        """
        self.message = message
        # Call parent constructor with message to ensure args[0] is set
        super().__init__(message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message

    def __repr__(self) -> str:
        """Return a string representation of the exception."""
        return f"{self.__class__.__name__}({self.message!r})"
