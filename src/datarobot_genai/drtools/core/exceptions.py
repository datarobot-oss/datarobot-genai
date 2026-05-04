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

from enum import StrEnum


class ToolErrorKind(StrEnum):
    """High-level category for tool failures (logging and MCP-friendly).

    ``SCHEMA`` — tool call arguments do not match the tool input schema (MCP/JSON shape, types,
    required fields). Use ``VALIDATION`` for domain rules enforced inside the tool body.
    """

    SCHEMA = "schema"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    NOT_FOUND = "not_found"
    UPSTREAM = "upstream"
    INTERNAL = "internal"


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
        kind: Semantic category of the failure (defaults to INTERNAL). Use ``ToolErrorKind.SCHEMA``
            when MCP tool-call JSON or types do not match the declared input schema.
        args: Tuple containing the error message (for compatibility with base Exception).

    Examples
    --------
        >>> raise ToolError("Failed to create deployment: No prediction servers available")
        >>> raise ToolError(f"Job failed with status {job.status}", kind=ToolErrorKind.UPSTREAM)
        >>> try:
        ...     risky_operation()
        ... except ValueError as e:
        ...     raise ToolError(f"Invalid input: {e}", kind=ToolErrorKind.VALIDATION)
        >>> raise ToolError("Arguments do not match tool schema", kind=ToolErrorKind.SCHEMA)
    """

    def __init__(self, message: str, *, kind: ToolErrorKind = ToolErrorKind.INTERNAL) -> None:
        """Initialize the ToolError with a descriptive message.

        Args:
            message: A clear, user-friendly description of what went wrong.
                This message will be displayed to the end user.
            kind: Category used for routing, metrics, and MCP error strings.
        """
        self.message = message
        self.kind = kind
        super().__init__(message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message

    def __repr__(self) -> str:
        """Return a string representation of the exception."""
        return f"{self.__class__.__name__}({self.message!r}, kind={self.kind!r})"
