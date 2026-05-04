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

import functools
import logging
import re
import traceback
from collections.abc import Callable
from typing import Any
from typing import TypeVar

from fastmcp.exceptions import ToolError as FastMCPToolError
from fastmcp.exceptions import ValidationError as FastMCPValidationError
from pydantic import ValidationError as PydanticValidationError

from datarobot_genai.drtools.core.exceptions import ToolError as DRToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind

# Secret patterns to redact from logs
SECRET_PATTERNS = [
    r"([a-zA-Z0-9]{20,})",  # Long alphanumeric strings (potential tokens)
    r"(sk-[a-zA-Z0-9]{48})",  # OpenAI-style keys
    r"(AKIA[0-9A-Z]{16})",  # AWS Access Key pattern
]


class SecretRedactingFormatter(logging.Formatter):
    """Custom formatter that redacts sensitive information from logs."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        return self._redact_secrets(msg)

    def _redact_secrets(self, message: str) -> str:
        """Redact potential secrets from log messages."""
        for pattern in SECRET_PATTERNS:
            message = re.sub(pattern, "[REDACTED]", message)
        return message


class MCPLogging:
    """MCP Logging class."""

    def __init__(self, level: str = "INFO") -> None:
        """Initialize the MCP logging."""
        self._level = level
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging with secret redaction and set log level."""
        # Remove all existing handlers
        logging.root.handlers.clear()

        # Add a console handler with our formatter
        handler = logging.StreamHandler()
        logger_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = SecretRedactingFormatter(logger_format)
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)
        logging.root.setLevel(self._level)


# Type variable for generic function type
F = TypeVar("F", bound=Callable[..., Any])


def _log_error(logger: logging.Logger, func_name: str, error: Exception, **kwargs: Any) -> str:
    """Log errors in a consistent format."""
    error_msg = f"{type(error).__name__}: {str(error)}"
    logger.error(f"Error in {func_name}: {error_msg}")
    logger.debug(f"Full traceback: {traceback.format_exc()}")
    logger.debug(f"Function arguments: {kwargs}")
    return f"Error in {func_name}: {error_msg}"


def _mcp_message_for_dr_tool_error(e: DRToolError) -> str:
    """Format drtools ToolError for fastmcp (kind prefix, no duplicate wrapping)."""
    return f"[{e.kind.value}] {e.message}"


def _kind_for_wrapped_exception(exc: Exception) -> ToolErrorKind:
    """Map platform/SDK HTTP errors to kinds when wrapping unknown exceptions.

    Used only when the tool did not raise :class:`DRToolError` — e.g. other tools, deep SDK
    calls, or new code paths. Predictive tools should still prefer explicit
    ``raise_tool_error_for_client_error`` where :class:`~datarobot.errors.ClientError` is expected.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status == 404:
            return ToolErrorKind.NOT_FOUND
        if status in (401, 403):
            return ToolErrorKind.AUTHENTICATION
        if 400 <= status < 600:
            return ToolErrorKind.UPSTREAM
    lowered = str(exc).lower()
    if "404" in lowered and ("not found" in lowered or "does not exist" in lowered):
        return ToolErrorKind.NOT_FOUND
    return ToolErrorKind.INTERNAL


def log_execution(func: F) -> F:
    """Log execution with error handling."""
    logger = logging.getLogger(func.__module__)

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            logger.info(f"Starting {func.__name__}")
            logger.debug(f"Arguments: {args}, {kwargs}")
            result = await func(*args, **kwargs)
            logger.info(f"Completed {func.__name__}")
            return result
        except FastMCPToolError as e:
            _log_error(logger, func.__name__, e, args=args, kwargs=kwargs)
            raise
        except DRToolError as e:
            _log_error(logger, func.__name__, e, args=args, kwargs=kwargs)
            raise FastMCPToolError(_mcp_message_for_dr_tool_error(e)) from e
        except (PydanticValidationError, FastMCPValidationError) as e:
            error_msg = _log_error(logger, func.__name__, e, args=args, kwargs=kwargs)
            raise FastMCPToolError(f"[{ToolErrorKind.SCHEMA.value}] {error_msg}") from e
        except Exception as e:
            error_msg = _log_error(logger, func.__name__, e, args=args, kwargs=kwargs)
            kind = _kind_for_wrapped_exception(e)
            raise FastMCPToolError(f"[{kind.value}] {error_msg}") from e

    return wrapper  # type: ignore[return-value]
