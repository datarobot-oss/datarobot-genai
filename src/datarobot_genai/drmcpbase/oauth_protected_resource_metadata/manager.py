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
    """How a client may present its bearer token to this resource server.

    RFC 6750 defines three ways a client may present a bearer token: the
    Authorization header, a form-encoded body parameter, and a URI query
    parameter. RFC 9728 lets the resource server declare which of them it
    actually accepts, so a client doesn't have to guess.

    We accept the header and nothing else. The other two are discouraged by RFC
    6750 itself — a query-string token ends up in access logs, proxy logs,
    Referer headers and browser history; the body form constrains the content
    type.

    This is a declaration, not an enforcement point: nothing reads it inbound,
    and the request side only ever looks at ``OAUTH_TOKEN_HEADERS``. Adding a
    member here would change what is advertised without changing what is
    accepted.
    """

    HEADER = auto()

    def get_name_in_lower_case(self) -> str:
        return self.name.lower()

    @classmethod
    def get_complete_list_of_supported_methods(cls) -> list[str]:
        return [supported_method.get_name_in_lower_case() for supported_method in cls]


class MCPOAuthProtectedResourceMetadataManager:
    def __init__(
        self, metadata_config: MCPOAuthProtectedResourceMetadataConfig | None = None
    ) -> None:
        self._metadata_config = metadata_config

    def load_config(self) -> MCPOAuthProtectedResourceMetadataConfig | None:
        """Return the configured metadata, or ``None`` when nothing was set."""
        if self._metadata_config is None or self._metadata_config.is_empty():
            return None
        return self._metadata_config

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
        return metadata.to_dict_without_null_attribute()
