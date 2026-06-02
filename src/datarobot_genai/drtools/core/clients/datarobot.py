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
from collections.abc import Generator
from contextlib import contextmanager

from datarobot.client import client_configuration
from datarobot.context import Context as DRContext

from datarobot_genai.drtools.core.auth import resolve_token_from_headers
from datarobot_genai.drtools.core.credentials import get_credentials
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind

logger = logging.getLogger(__name__)


def get_datarobot_access_token() -> str:
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


class ThreadSafeDataRobotClient:
    def __init__(self) -> None:
        self.endpoint = get_credentials().datarobot.endpoint

    @contextmanager
    def get_client_context_with_token_from_request_header(self) -> Generator[None, None, None]:
        token = get_datarobot_access_token()
        with client_configuration(token=token, endpoint=self.endpoint):
            # Avoid use-case context from trafaret affecting tool calls.
            DRContext.use_case = None
            yield
