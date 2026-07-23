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
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class BaseDataClass:
    def to_json_without_null_attribute(self) -> dict[str, Any]:
        return asdict(
            self,  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType]
            dict_factory=lambda x: {k: v for k, v in x if v is not None},
        )

    def to_json_string(self) -> str:
        return json.dumps(self.to_json_without_null_attribute())


@dataclass
class XAATokenExchangeParams(BaseDataClass):
    trusted_issuer: str
    audience: str

    @classmethod
    def from_json(cls, json_dict: dict[str, str]) -> "XAATokenExchangeParams":
        return cls(json_dict["trusted_issuer"], json_dict["audience"])


@dataclass
class XAATokenRequestParams(BaseDataClass):
    token_url: str
    # audience can be None if it is not setup for AuthN & AuthZ check (as resource) in IdP.
    audience: str | None
    scopes: list[str]

    @classmethod
    def from_json(cls, json_dict: dict[str, Any]) -> "XAATokenRequestParams":
        return cls(json_dict["token_url"], json_dict.get("audience"), json_dict["scopes"])


@dataclass
class XAAMetadata(BaseDataClass):
    token_endpoint_auth_method: str
    token_exchange: XAATokenExchangeParams
    token_request: XAATokenRequestParams

    @classmethod
    def from_json(cls, metadata_in_json: dict[str, Any]) -> "XAAMetadata":
        return cls(
            metadata_in_json["token_endpoint_auth_method"],
            XAATokenExchangeParams.from_json(metadata_in_json["token_exchange"]),
            XAATokenRequestParams.from_json(metadata_in_json["token_request"]),
        )


@dataclass
class MCPOAuthProtectedResourceMetadataConfig(BaseDataClass):
    resource: str
    authorization_servers: list[str]
    scopes_supported: list[str]
    xaa_metadata: XAAMetadata | None

    @classmethod
    def from_json(
        cls, metadata_in_json: dict[str, Any]
    ) -> "MCPOAuthProtectedResourceMetadataConfig":
        xaa_metadata = (
            XAAMetadata.from_json(metadata_in_json["xaa_metadata"])
            if metadata_in_json.get("xaa_metadata")
            else None
        )
        return cls(
            metadata_in_json["resource"],
            metadata_in_json["authorization_servers"],
            metadata_in_json["scopes_supported"],
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
