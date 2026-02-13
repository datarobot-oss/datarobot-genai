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

"""DataRobot API client and utilities for tools (e.g. predictive tools)."""

import logging
from typing import Any

import datarobot as dr
from datarobot.context import Context as DRContext
from fastmcp.exceptions import ToolError

from datarobot_genai.drmcp.core.clients import resolve_token_from_headers
from datarobot_genai.drmcp.core.credentials import get_credentials

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
            "Please provide it via 'Authorization' (Bearer), 'x-datarobot-api-token' headers."
        )
    return token


class DataRobotClient:
    """Client for interacting with DataRobot API in tools.

    Wraps the DataRobot Python SDK (datarobot package). Obtain the token
    via get_datarobot_access_token() and pass it to the constructor.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    def get_client(self) -> Any:
        """
        Configure the DataRobot SDK with this client's token and return the dr module.

        The returned value is the global datarobot module (dr) after
        dr.Client(token=..., endpoint=...) has been called. Use it as
        client.Project.list(), client.Deployment.get(...), etc.

        Returns
        -------
            The datarobot module (dr) configured for the current token and endpoint.
        """
        credentials = get_credentials()
        dr.Client(token=self._token, endpoint=credentials.datarobot.endpoint)
        # Avoid use-case context from trafaret affecting tool calls
        DRContext.use_case = None
        return dr
