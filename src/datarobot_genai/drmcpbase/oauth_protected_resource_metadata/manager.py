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
import json
import logging
import os
from enum import Enum
from enum import auto
from typing import Any

from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    MCPOAuthProtectedResourceMetadata,
)
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    MCPOAuthProtectedResourceMetadataAdminConfig,
)
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    MCPOAuthProtectedResourceMetadataConfig,
)

logger = logging.getLogger(__name__)


class SupportedMethodsToSendBearerToken(Enum):
    HEADER = auto()

    def get_name_in_lower_case(self) -> str:
        return self.name.lower()

    @classmethod
    def get_complete_list_of_supported_methods(cls) -> list[str]:
        return [supported_method.get_name_in_lower_case() for supported_method in cls]


class ContainerEnvVar(Enum):
    MCP_OAUTH_PROTECTED_RESOURCE_METADATA = auto()

    def get_env_var_value(self) -> str | None:
        return os.getenv(self.name)


class MCPOAuthProtectedResourceMetadataManager:
    @staticmethod
    def load_config() -> MCPOAuthProtectedResourceMetadataConfig | None:
        metadata_in_json_str = (
            ContainerEnvVar.MCP_OAUTH_PROTECTED_RESOURCE_METADATA.get_env_var_value()
        )
        if metadata_in_json_str:
            metadata_in_json = json.loads(metadata_in_json_str)
            return MCPOAuthProtectedResourceMetadataConfig.from_json(metadata_in_json)
        else:
            return None

    @staticmethod
    def get_admin_config() -> MCPOAuthProtectedResourceMetadataAdminConfig:
        return MCPOAuthProtectedResourceMetadataAdminConfig(
            [SupportedMethodsToSendBearerToken.HEADER.get_name_in_lower_case()]
        )

    def get_protected_resource_metadata(self) -> MCPOAuthProtectedResourceMetadata | None:
        metadata_config = self.load_config()
        if not metadata_config:
            return None
        admin_config = self.get_admin_config()
        return MCPOAuthProtectedResourceMetadata.build(metadata_config, admin_config)

    def get_protected_resource_metadata_api_response(self) -> dict[str, Any] | None:
        metadata = self.get_protected_resource_metadata()
        if not metadata:
            return None
        return metadata.to_json_without_null_attribute()
