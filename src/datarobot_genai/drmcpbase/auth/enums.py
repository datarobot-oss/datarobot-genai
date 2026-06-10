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
import logging
from enum import Enum
from enum import auto

from fastmcp.server.dependencies import get_http_headers

from datarobot_genai.drmcpbase.auth.exceptions import InvalidBearerTokenError
from datarobot_genai.drmcpbase.auth.exceptions import (
    NoDataRobotBearerTokenFoundInRequestContextError,
)
from datarobot_genai.drmcpbase.auth.exceptions import NoHeadersFoundInRequestContextError

logger = logging.getLogger(__name__)


class DataRobotBearerHeaderEnum(Enum):
    X_DATAROBOT_AUTHORIZATION = auto()
    AUTHORIZATION = auto()

    def get_normalized_header_key(self) -> str:
        return self.name.lower().replace("_", "-")

    @staticmethod
    def extract_bearer_token_value(bearer_token_string: str) -> str:
        bearer_token_parts = bearer_token_string.split("Bearer ")
        if len(bearer_token_parts) != 2:
            raise InvalidBearerTokenError(
                "Invalid Bearer token value. It should start with Bearer ."
            )
        return bearer_token_parts[1]

    def get_from_mcp_request(self) -> str:
        headers = get_http_headers(include_all=True)
        if not headers:
            raise NoHeadersFoundInRequestContextError("No header is found in MCP request context.")

        if self.get_normalized_header_key() not in headers:
            raise NoDataRobotBearerTokenFoundInRequestContextError(
                "No bearer token is found in MCP request context."
            )

        return self.extract_bearer_token_value(headers[self.get_normalized_header_key()])
