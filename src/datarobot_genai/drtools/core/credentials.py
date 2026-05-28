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

from typing import Any

from pydantic import AliasChoices
from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from .config_utils import extract_datarobot_runtime_param_payload
from .constants import DEFAULT_DATAROBOT_ENDPOINT
from .constants import RUNTIME_PARAM_ENV_VAR_NAME_PREFIX


class DataRobotCredentials(BaseSettings):
    """DataRobot API credentials."""

    application_api_token: str = Field(
        default="",
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "DATAROBOT_API_TOKEN",
            "DATAROBOT_API_TOKEN",
        ),
        description="DataRobot API token",
    )
    endpoint: str = Field(
        default=DEFAULT_DATAROBOT_ENDPOINT,
        validation_alias=AliasChoices(
            RUNTIME_PARAM_ENV_VAR_NAME_PREFIX + "DATAROBOT_ENDPOINT",
            "DATAROBOT_ENDPOINT",
        ),
        description="DataRobot API endpoint",
    )

    @field_validator(
        "application_api_token",
        "endpoint",
        mode="before",
    )
    @classmethod
    def validate_runtime_params(cls, v: Any) -> Any:
        """Validate runtime parameters."""
        return extract_datarobot_runtime_param_payload(v)

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_file_encoding="utf-8",
        extra="ignore",
    )


class MCPServerCredentials(BaseSettings):
    """Application credentials for MCP server."""

    datarobot: DataRobotCredentials = Field(default_factory=DataRobotCredentials)

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def has_datarobot_credentials(self) -> bool:
        """Check if DataRobot credentials are configured."""
        return bool(self.datarobot.application_api_token)


# Global credentials instance
_credentials: MCPServerCredentials | None = None


def get_credentials() -> MCPServerCredentials:
    """Get the global credentials instance."""
    # Use a local variable to avoid global statement warning
    credentials = _credentials
    if credentials is None:
        credentials = MCPServerCredentials()
        # Update the global variable
        globals()["_credentials"] = credentials
    return credentials
