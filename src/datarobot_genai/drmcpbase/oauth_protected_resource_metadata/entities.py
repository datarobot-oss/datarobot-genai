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
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Only ``private_key_jwt`` is implemented today, so the config may omit it.
DEFAULT_TOKEN_ENDPOINT_AUTH_METHOD = "private_key_jwt"

# Config key for the Cross-Application Access block. In the *served* document it
# becomes ``x_cross_application_access``: members that are not registered RFC 9728
# parameters carry an ``x_`` prefix so clients can tell them apart.
CROSS_APPLICATION_ACCESS_CONFIG_KEY = "cross_application_access"


class BaseDataClass:
    def to_dict_without_null_attribute(self) -> dict[str, Any]:
        return asdict(
            self,  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType]
            dict_factory=lambda x: {k: v for k, v in x if v is not None},
        )

    def to_yaml_string(self) -> str:
        return yaml.safe_dump(self.to_dict_without_null_attribute())


@dataclass
class XAATokenExchangeParams(BaseDataClass):
    trusted_issuer: str
    audience: str

    @classmethod
    def from_dict(cls, dict_input: dict[str, str]) -> "XAATokenExchangeParams":
        return cls(dict_input["trusted_issuer"], dict_input["audience"])


@dataclass
class XAATokenRequestParams(BaseDataClass):
    token_url: str
    # audience can be None if it is not setup for AuthN & AuthZ check (as resource) in IdP.
    audience: str | None
    scopes: list[str]

    @classmethod
    def from_dict(cls, dict_input: dict[str, Any]) -> "XAATokenRequestParams":
        return cls(dict_input["token_url"], dict_input.get("audience"), dict_input["scopes"])


@dataclass
class CrossApplicationAccessMetadata(BaseDataClass):
    """Cross-Application Access parameters for the hybrid RFC 8693 / RFC 7523 flow.

    Mirrors ``dragent.cross_app_access_config.CrossApplicationAccessConfig`` so an
    agent can either declare the block itself or read this one off the MCP
    server's protected resource metadata document.
    """

    token_exchange: XAATokenExchangeParams
    token_request: XAATokenRequestParams
    token_endpoint_auth_method: str = DEFAULT_TOKEN_ENDPOINT_AUTH_METHOD

    @classmethod
    def from_dict(cls, metadata_in_dict: dict[str, Any]) -> "CrossApplicationAccessMetadata":
        return cls(
            token_exchange=XAATokenExchangeParams.from_dict(metadata_in_dict["token_exchange"]),
            token_request=XAATokenRequestParams.from_dict(metadata_in_dict["token_request"]),
            token_endpoint_auth_method=metadata_in_dict.get(
                "token_endpoint_auth_method", DEFAULT_TOKEN_ENDPOINT_AUTH_METHOD
            ),
        )


@dataclass
class MCPOAuthProtectedResourceMetadataConfig(BaseDataClass):
    """User-authored config (``dr_mcp/oauth-config.yaml`` / ``MCP_OAUTH_METADATA``).

    Every field is optional: ``resource``, ``authorization_servers`` and
    ``scopes_supported`` are published verbatim but nothing enforces them yet, so
    a config that only declares ``cross_application_access`` is valid. Unknown
    keys are ignored, so a config still using the pre-rename ``xaa_metadata``
    block silently publishes no Cross-Application Access metadata.
    """

    resource: str | None = None
    authorization_servers: list[str] | None = None
    scopes_supported: list[str] | None = None
    cross_application_access: CrossApplicationAccessMetadata | None = None
    mcp_enable_unauthenticated_well_known_route: bool | None = None

    @classmethod
    def from_dict(
        cls, metadata_in_dict: dict[str, Any]
    ) -> "MCPOAuthProtectedResourceMetadataConfig":
        cross_application_access_in_dict = metadata_in_dict.get(CROSS_APPLICATION_ACCESS_CONFIG_KEY)
        cross_application_access = (
            CrossApplicationAccessMetadata.from_dict(cross_application_access_in_dict)
            if cross_application_access_in_dict
            else None
        )
        return cls(
            resource=metadata_in_dict.get("resource"),
            authorization_servers=metadata_in_dict.get("authorization_servers"),
            scopes_supported=metadata_in_dict.get("scopes_supported"),
            cross_application_access=cross_application_access,
            mcp_enable_unauthenticated_well_known_route=metadata_in_dict.get(
                "mcp_enable_unauthenticated_well_known_route"
            ),
        )


@dataclass
class MCPOAuthProtectedResourceMetadataAdminConfig(BaseDataClass):
    bearer_methods_supported: list[str]


@dataclass
class MCPOAuthProtectedResourceMetadata(BaseDataClass):
    """The document served at ``/.well-known/oauth-protected-resource``.

    Registered RFC 9728 parameters keep their standard names; everything
    DataRobot-specific is ``x_``-prefixed. Unset fields are dropped by
    ``to_dict_without_null_attribute``, so a config that only declares
    ``cross_application_access`` yields just the XAA block plus
    ``bearer_methods_supported``.
    """

    bearer_methods_supported: list[str]
    resource: str | None = None
    authorization_servers: list[str] | None = None
    scopes_supported: list[str] | None = None
    x_cross_application_access: CrossApplicationAccessMetadata | None = None
    x_mcp_enable_unauthenticated_well_known_route: bool | None = None

    @classmethod
    def build(
        cls,
        user_config: MCPOAuthProtectedResourceMetadataConfig,
        admin_config: MCPOAuthProtectedResourceMetadataAdminConfig,
    ) -> "MCPOAuthProtectedResourceMetadata":
        return cls(
            bearer_methods_supported=admin_config.bearer_methods_supported,
            resource=user_config.resource,
            authorization_servers=user_config.authorization_servers,
            scopes_supported=user_config.scopes_supported,
            x_cross_application_access=user_config.cross_application_access,
            x_mcp_enable_unauthenticated_well_known_route=(
                user_config.mcp_enable_unauthenticated_well_known_route
            ),
        )
