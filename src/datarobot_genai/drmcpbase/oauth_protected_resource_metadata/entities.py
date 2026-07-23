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
class XAAMetadata(BaseDataClass):
    token_endpoint_auth_method: str
    token_exchange: XAATokenExchangeParams
    token_request: XAATokenRequestParams

    @classmethod
    def from_dict(cls, metadata_in_dict: dict[str, Any]) -> "XAAMetadata":
        return cls(
            metadata_in_dict["token_endpoint_auth_method"],
            XAATokenExchangeParams.from_dict(metadata_in_dict["token_exchange"]),
            XAATokenRequestParams.from_dict(metadata_in_dict["token_request"]),
        )


@dataclass
class MCPOAuthProtectedResourceMetadataConfig(BaseDataClass):
    resource: str
    authorization_servers: list[str]
    scopes_supported: list[str]
    xaa_metadata: XAAMetadata | None

    @classmethod
    def from_dict(
        cls, metadata_in_dict: dict[str, Any]
    ) -> "MCPOAuthProtectedResourceMetadataConfig":
        xaa_metadata = (
            XAAMetadata.from_dict(metadata_in_dict["xaa_metadata"])
            if metadata_in_dict.get("xaa_metadata")
            else None
        )
        return cls(
            metadata_in_dict["resource"],
            metadata_in_dict["authorization_servers"],
            metadata_in_dict["scopes_supported"],
            xaa_metadata,
        )


@dataclass
class MCPOAuthProtectedResourceMetadataAdminConfig(BaseDataClass):
    bearer_methods_supported: list[str]


@dataclass
class MCPOAuthProtectedResourceMetadata(BaseDataClass):
    resource: str
    authorization_servers: list[str]
    bearer_methods_supported: list[str]
    scopes_supported: list[str]
    xaa_metadata: XAAMetadata | None

    @classmethod
    def build(
        cls,
        user_config: MCPOAuthProtectedResourceMetadataConfig,
        admin_config: MCPOAuthProtectedResourceMetadataAdminConfig,
    ) -> "MCPOAuthProtectedResourceMetadata":
        return cls(
            user_config.resource,
            user_config.authorization_servers,
            admin_config.bearer_methods_supported,
            user_config.scopes_supported,
            user_config.xaa_metadata,
        )
