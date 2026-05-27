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

"""DataRobot API client for tools."""

import logging
from collections.abc import AsyncGenerator
from collections.abc import Generator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from typing import Any

import datarobot as dr
from datarobot.client import client_configuration
from datarobot.context import Context as DRContext

from datarobot_genai.drtools.core.auth import resolve_token_from_headers
from datarobot_genai.drtools.core.credentials import get_credentials
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind

logger = logging.getLogger(__name__)


async def get_datarobot_access_token() -> str:
    """
    Get DataRobot API token from HTTP headers.

    Uses the same token extraction as core (auth headers and authorization
    context metadata). For use in tools only; core modules use get_sdk_client()
    from drmcp.core.clients.

    Returns
    -------
        API token string

    Raises
    ------
        ToolError: If no API token is found in headers
    """
    token = resolve_token_from_headers()
    if not token:
        logger.warning("DataRobot API token not found in headers")
        raise ToolError(
            "DataRobot API token not found in headers. "
            "Please provide it via 'Authorization' (Bearer), 'x-datarobot-api-token' headers.",
            kind=ToolErrorKind.AUTHENTICATION,
        )
    return token


class DataRobotClient:
    """Client for interacting with DataRobot API in tools.

    Wraps the DataRobot Python SDK (datarobot package). Obtain the token
    via get_datarobot_access_token() and pass it to the constructor.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    @contextmanager
    def get_client(self) -> Generator[Any, None, None]:
        """
        Context manager that configures the DR SDK for this asyncio task and yields dr.

        Uses client_configuration() which stores the RESTClientObject in a ContextVar
        (_context_client) rather than a plain global. Each asyncio Task that enters this
        context gets its own isolated client — preventing token mixing between concurrent
        MCP tool invocations.
        """
        credentials = get_credentials()
        with client_configuration(token=self._token, endpoint=credentials.datarobot.endpoint):
            # Avoid use-case context from trafaret affecting tool calls.
            DRContext.use_case = None
            yield dr


@asynccontextmanager
async def dr_client() -> AsyncGenerator[Any, None]:
    """Async context manager that resolves the request token and yields a configured dr module.

    Combines get_datarobot_access_token() + DataRobotClient.get_client() into a single
    entry point, eliminating the two-line boilerplate repeated across every tool function.

    Usage::

        async with dr_client() as client:
            deployments = client.Deployment.list()
    """
    token = await get_datarobot_access_token()
    with DataRobotClient(token).get_client() as client:
        yield client
