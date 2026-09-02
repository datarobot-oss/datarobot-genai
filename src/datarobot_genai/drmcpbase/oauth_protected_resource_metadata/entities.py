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
from enum import Enum
from enum import auto
from typing import Any

logger = logging.getLogger(__name__)

# Only ``private_key_jwt`` is implemented today, so the config may omit it.
DEFAULT_TOKEN_ENDPOINT_AUTH_METHOD = "private_key_jwt"


def split_list_setting(value: str | None) -> list[str] | None:
    """Read a comma-separated setting, treating blank and unset alike."""
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()] or None


class BaseDataClass:
    def to_dict_without_null_attribute(self) -> dict[str, Any]:
        return asdict(
            self,  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType]
            dict_factory=lambda x: {k: v for k, v in x if v is not None},
        )


@dataclass
class XAATokenExchangeParams(BaseDataClass):
    trusted_issuer: str
    audience: str


@dataclass
class XAATokenRequestParams(BaseDataClass):
    token_url: str
    # audience can be None if it is not setup for AuthN & AuthZ check (as resource) in IdP.
    audience: str | None
    scopes: list[str]


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
    def from_settings(
        cls,
        *,
        trusted_issuer: str | None = None,
        exchange_audience: str | None = None,
        token_url: str | None = None,
        token_audience: str | None = None,
        scopes: str | None = None,
        token_endpoint_auth_method: str | None = None,
    ) -> "CrossApplicationAccessMetadata | None":
        """Assemble the block from the flat ``MCP_XAA_*`` settings.

        All four required settings must be present. A partial block is dropped
        with a warning naming exactly what is missing rather than raising: the
        deployment tooling rejects it up front, so reaching here means the
        container was configured by hand and a broken well-known route helps
        nobody.
        """
        parsed_scopes = split_list_setting(scopes)
        # Env var names keyed to their values: both guards and the warning are
        # derived from this one mapping, so the set of required settings can
        # never drift from the message that names them.
        required: dict[str, str | list[str] | None] = {
            "MCP_XAA_TRUSTED_ISSUER": trusted_issuer,
            "MCP_XAA_EXCHANGE_AUDIENCE": exchange_audience,
            "MCP_XAA_TOKEN_URL": token_url,
            "MCP_XAA_SCOPES": parsed_scopes,
        }
        if not any(required.values()):
            return None
        if missing := [name for name, value in required.items() if not value]:
            logger.warning(
                "Incomplete Cross-Application Access settings; publishing no "
                "cross_application_access block. Missing: %s.",
                ", ".join(missing),
            )
            return None
        # `missing` being empty proves these are set; spelled out only because
        # mypy cannot see that through the mapping.
        assert trusted_issuer and exchange_audience and token_url and parsed_scopes
        return cls(
            token_exchange=XAATokenExchangeParams(
                trusted_issuer=trusted_issuer,
                audience=exchange_audience,
            ),
            token_request=XAATokenRequestParams(
                token_url=token_url,
                audience=token_audience,
                scopes=parsed_scopes,
            ),
            token_endpoint_auth_method=(
                token_endpoint_auth_method or DEFAULT_TOKEN_ENDPOINT_AUTH_METHOD
            ),
        )


@dataclass
class MCPOAuthProtectedResourceMetadataConfig(BaseDataClass):
    """User-authored part of the document, from the server's own settings.

    Every field is optional, so a config that only declares
    ``cross_application_access`` is valid. ``resource`` and
    ``authorization_servers`` are published verbatim; ``scopes_supported`` is
    handed in by the server, derived from the scope rules it actually enforces.
    """

    resource: str | None = None
    authorization_servers: list[str] | None = None
    scopes_supported: list[str] | None = None
    cross_application_access: CrossApplicationAccessMetadata | None = None

    @classmethod
    def from_settings(
        cls,
        *,
        resource: str | None = None,
        authorization_servers: str | None = None,
        scopes_supported: list[str] | None = None,
        xaa_trusted_issuer: str | None = None,
        xaa_exchange_audience: str | None = None,
        xaa_token_url: str | None = None,
        xaa_token_audience: str | None = None,
        xaa_scopes: str | None = None,
        xaa_token_endpoint_auth_method: str | None = None,
    ) -> "MCPOAuthProtectedResourceMetadataConfig":
        """Assemble the config from the server's settings.

        Every parameter is the flat string an env var or runtime parameter can
        carry — except ``scopes_supported``, which takes a list: its only
        source is the scope rules declared in code and configuration (see
        ``datarobot_genai.drmcpbase.oauth_scopes.derived_scopes``), never a
        flat setting, so there is no comma-string for callers to hand-join
        here or for this to split. Blank entries are dropped and an empty list
        publishes nothing, like every other unset field.
        """
        scopes = [scope.strip() for scope in (scopes_supported or []) if scope.strip()]
        return cls(
            resource=resource or None,
            authorization_servers=split_list_setting(authorization_servers),
            scopes_supported=scopes or None,
            cross_application_access=CrossApplicationAccessMetadata.from_settings(
                trusted_issuer=xaa_trusted_issuer,
                exchange_audience=xaa_exchange_audience,
                token_url=xaa_token_url,
                token_audience=xaa_token_audience,
                scopes=xaa_scopes,
                token_endpoint_auth_method=xaa_token_endpoint_auth_method,
            ),
        )

    def is_empty(self) -> bool:
        return not self.to_dict_without_null_attribute()


@dataclass
class MCPOAuthProtectedResourceMetadataAdminConfig(BaseDataClass):
    """Server-owned facts about this deployment, not user-authored metadata."""

    bearer_methods_supported: list[str]


@dataclass
class MCPOAuthProtectedResourceMetadata(BaseDataClass):
    """The document served at ``/.well-known/oauth-protected-resource``.

    Registered RFC 9728 parameters keep their standard names.
    ``cross_application_access`` is a DataRobot addition published under its own
    name, matching the agent-side config block exactly. Unset fields are dropped
    by ``to_dict_without_null_attribute``, so a config that only declares
    ``cross_application_access`` yields just that block plus
    ``bearer_methods_supported``.
    """

    bearer_methods_supported: list[str]
    resource: str | None = None
    authorization_servers: list[str] | None = None
    scopes_supported: list[str] | None = None
    cross_application_access: CrossApplicationAccessMetadata | None = None

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
            cross_application_access=user_config.cross_application_access,
        )


class ErrorCodeInAuthErrorResponse(Enum):
    """It is based on https://www.rfc-editor.org/info/rfc6750/#section-3.1."""

    INVALID_REQUEST = auto()
    INVALID_TOKEN = auto()
    INSUFFICIENT_SCOPE = auto()
    UNKNOWN = auto()  # This is not the part of the spec. I added here for better code readability.

    def to_value(self) -> str:
        return self.name.lower()


@dataclass
class AuthErrorResponse:
    resource_metadata: str
    error_code: ErrorCodeInAuthErrorResponse = ErrorCodeInAuthErrorResponse.UNKNOWN
    scopes: list[str] | None = None
    error_description: str | None = None

    def get_scopes_string_representation(self) -> str | None:
        if self.scopes:
            return " ".join(self.scopes)
        return None

    @staticmethod
    def get_header_name() -> str:
        return "WWW-Authenticate"

    def to_header_value(self) -> str:
        parts = [f'resource_metadata="{self.resource_metadata}"']
        if self.scopes:
            parts.append(f'scope="{self.get_scopes_string_representation()}"')
        if self.error_code != ErrorCodeInAuthErrorResponse.UNKNOWN:
            parts.append(f'error="{self.error_code.to_value()}"')
        if self.error_description:
            parts.append(f'error_description="{self.error_description}"')
        return "Bearer " + ", ".join(parts)

    def to_response_content(self) -> dict[str, str | None]:
        return {"error": self.error_code.to_value(), "error_description": self.error_description}
