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

from enum import StrEnum

from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from pydantic import Field

from .constants import DEFAULT_DATAROBOT_ENDPOINT


class AuthResolutionStrategy(StrEnum):
    """How tool authentication secrets are resolved."""

    HTTP = "http"
    CONFIG = "config"


class DataRobotCredentials(DataRobotAppFrameworkBaseSettings):
    """DataRobot API credentials.

    Loads from env vars (including MLOPS_RUNTIME_PARAM_*), .env, file secrets,
    and pulumi_config.json. Fields map by name: ``datarobot_api_token`` reads
    ``DATAROBOT_API_TOKEN``, ``datarobot_endpoint`` reads ``DATAROBOT_ENDPOINT``.
    """

    datarobot_api_token: str = ""
    datarobot_endpoint: str = DEFAULT_DATAROBOT_ENDPOINT


class ToolsAuthCredentials(DataRobotAppFrameworkBaseSettings):
    """Application credentials and auth resolution settings for tools.

    Config fields are used when ``auth_resolution_strategy`` is ``config``.
    See ``docs/drtools/auth.md`` for strategy details.
    """

    auth_resolution_strategy: AuthResolutionStrategy = AuthResolutionStrategy.HTTP
    datarobot: DataRobotCredentials = Field(default_factory=DataRobotCredentials)
    tavily_api_key: str = ""
    perplexity_api_key: str = ""
    atlassian_api_token: str = ""
    atlassian_email: str = ""
    atlassian_site_url: str = ""

    def has_datarobot_credentials(self) -> bool:
        """Check if DataRobot credentials are configured."""
        return bool(self.datarobot.datarobot_api_token)


# Global credentials instance
_credentials: ToolsAuthCredentials | None = None


def get_credentials() -> ToolsAuthCredentials:
    """Get the global credentials instance."""
    # Use a local variable to avoid global statement warning
    credentials = _credentials
    if credentials is None:
        credentials = ToolsAuthCredentials()
        # Update the global variable
        globals()["_credentials"] = credentials
    return credentials
