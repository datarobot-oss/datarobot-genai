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
from dataclasses import dataclass
from typing import Any

import pytest
import yaml

from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    DEFAULT_TOKEN_ENDPOINT_AUTH_METHOD,
)
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import BaseDataClass
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    CrossApplicationAccessMetadata,
)
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    MCPOAuthProtectedResourceMetadata,
)
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    MCPOAuthProtectedResourceMetadataAdminConfig,
)
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    MCPOAuthProtectedResourceMetadataConfig,
)
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    XAATokenExchangeParams,
)
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    XAATokenRequestParams,
)


@pytest.fixture
def mock_mcp_as_resource_server_url() -> str:
    return "https://foo/bar/mcp_resource_server"


@pytest.fixture
def mock_authorization_server_urls() -> list[str]:
    return ["https://foo/bar/authorization_server"]


@pytest.fixture
def mock_scopes_supported() -> list[str]:
    return ["scope"]


@pytest.fixture
def mock_token_endpoint_auth_method() -> str:
    return "private_key_jwt"


@pytest.fixture
def mock_token_exchange_trusted_issuer() -> str:
    return "https://foo/bar/issuer"


@pytest.fixture
def mock_token_exchange_audience() -> str:
    return "https://foo/bar/token_exchange_audience"


@pytest.fixture
def mock_token_request_token_url() -> str:
    return "https://foo/bar/token"


@pytest.fixture
def mock_token_request_audience() -> str:
    return "https://foo/bar/token_request_audience"


@pytest.fixture
def mock_token_request_scopes() -> list[str]:
    return ["scope"]


@pytest.fixture
def cross_application_access_in_dict(
    mock_token_endpoint_auth_method: str,
    mock_token_exchange_trusted_issuer: str,
    mock_token_exchange_audience: str,
    mock_token_request_token_url: str,
    mock_token_request_audience: str,
    mock_token_request_scopes: list[str],
) -> dict[str, Any]:
    return {
        "token_endpoint_auth_method": mock_token_endpoint_auth_method,
        "token_exchange": {
            "trusted_issuer": mock_token_exchange_trusted_issuer,
            "audience": mock_token_exchange_audience,
        },
        "token_request": {
            "token_url": mock_token_request_token_url,
            "audience": mock_token_request_audience,
            "scopes": mock_token_request_scopes,
        },
    }


@pytest.fixture
def metadata_in_dict(
    mock_mcp_as_resource_server_url: str,
    mock_authorization_server_urls: list[str],
    mock_scopes_supported: list[str],
    cross_application_access_in_dict: dict[str, Any],
) -> dict[str, Any]:
    return {
        "resource": mock_mcp_as_resource_server_url,
        "authorization_servers": mock_authorization_server_urls,
        "scopes_supported": mock_scopes_supported,
        "cross_application_access": cross_application_access_in_dict,
    }


@dataclass
class DummyDataClassInheritingBaseDataClass(BaseDataClass):
    attribute: int
    nullable_attribute: int | None


class TestBaseDataClass:
    def test_to_dict_without_null_attribute(self) -> None:
        dataclass_object = DummyDataClassInheritingBaseDataClass(1, None)
        assert dataclass_object.to_dict_without_null_attribute() == {"attribute": 1}

    def test_to_yaml_string(self) -> None:
        dataclass_object = DummyDataClassInheritingBaseDataClass(1, None)
        assert dataclass_object.to_yaml_string() == "attribute: 1\n"


class TestCrossApplicationAccessMetadata:
    @pytest.fixture
    def metadata_without_token_request_audience(
        self,
        cross_application_access_in_dict: dict[str, Any],
    ) -> dict[str, Any]:
        cross_application_access_in_dict["token_request"].pop("audience")
        return cross_application_access_in_dict

    def test_load_from_dict(
        self,
        cross_application_access_in_dict: dict[str, Any],
        mock_token_endpoint_auth_method: str,
        mock_token_exchange_trusted_issuer: str,
        mock_token_exchange_audience: str,
        mock_token_request_token_url: str,
        mock_token_request_audience: str,
        mock_token_request_scopes: list[str],
    ) -> None:
        metadata = CrossApplicationAccessMetadata.from_dict(cross_application_access_in_dict)

        assert isinstance(metadata, CrossApplicationAccessMetadata)
        assert metadata.token_endpoint_auth_method == mock_token_endpoint_auth_method
        token_exchange_params = metadata.token_exchange
        assert isinstance(token_exchange_params, XAATokenExchangeParams)
        assert token_exchange_params.trusted_issuer == mock_token_exchange_trusted_issuer
        assert token_exchange_params.audience == mock_token_exchange_audience
        token_request_params = metadata.token_request
        assert isinstance(token_request_params, XAATokenRequestParams)
        assert token_request_params.token_url == mock_token_request_token_url
        assert token_request_params.audience == mock_token_request_audience
        assert token_request_params.scopes == mock_token_request_scopes

    def test_load_from_dict_without_token_request_audience(
        self,
        metadata_without_token_request_audience: dict[str, Any],
    ) -> None:
        metadata = CrossApplicationAccessMetadata.from_dict(metadata_without_token_request_audience)

        assert isinstance(metadata, CrossApplicationAccessMetadata)
        assert metadata.token_request.audience is None

    def test_to_yaml_string(self, cross_application_access_in_dict: dict[str, Any]) -> None:
        metadata = CrossApplicationAccessMetadata.from_dict(cross_application_access_in_dict)
        assert metadata.to_yaml_string() == yaml.safe_dump(cross_application_access_in_dict)

    def test_token_endpoint_auth_method_defaults_when_omitted(
        self,
        cross_application_access_in_dict: dict[str, Any],
    ) -> None:
        cross_application_access_in_dict.pop("token_endpoint_auth_method")

        metadata = CrossApplicationAccessMetadata.from_dict(cross_application_access_in_dict)

        assert metadata.token_endpoint_auth_method == DEFAULT_TOKEN_ENDPOINT_AUTH_METHOD
        assert DEFAULT_TOKEN_ENDPOINT_AUTH_METHOD == "private_key_jwt"


class TestMCPOAuthProtectedResourceMetadataConfig:
    def test_load_from_dict(
        self,
        metadata_in_dict: dict[str, Any],
        mock_mcp_as_resource_server_url: str,
        mock_authorization_server_urls: list[str],
        mock_scopes_supported: list[str],
        cross_application_access_in_dict: dict[str, Any],
    ) -> None:
        metadata = MCPOAuthProtectedResourceMetadataConfig.from_dict(metadata_in_dict)

        assert metadata.resource == mock_mcp_as_resource_server_url
        assert metadata.authorization_servers == mock_authorization_server_urls
        assert metadata.scopes_supported == mock_scopes_supported
        assert isinstance(metadata.cross_application_access, CrossApplicationAccessMetadata)

    def test_to_yaml_string(self, metadata_in_dict: dict[str, Any]) -> None:
        metadata = MCPOAuthProtectedResourceMetadataConfig.from_dict(metadata_in_dict)
        assert metadata.to_yaml_string() == yaml.safe_dump(metadata_in_dict)

    def test_load_from_dict_with_only_cross_application_access(
        self,
        cross_application_access_in_dict: dict[str, Any],
    ) -> None:
        """resource/authorization_servers/scopes_supported have no logic behind them yet."""
        metadata = MCPOAuthProtectedResourceMetadataConfig.from_dict(
            {"cross_application_access": cross_application_access_in_dict}
        )

        assert metadata.resource is None
        assert metadata.authorization_servers is None
        assert metadata.scopes_supported is None
        assert metadata.mcp_enable_unauthenticated_well_known_route is None
        assert metadata.to_dict_without_null_attribute() == {
            "cross_application_access": cross_application_access_in_dict
        }

    def test_load_from_dict_with_empty_config(self) -> None:
        metadata = MCPOAuthProtectedResourceMetadataConfig.from_dict({})

        assert metadata.cross_application_access is None
        assert metadata.to_dict_without_null_attribute() == {}

    def test_load_from_dict_reads_unauthenticated_well_known_route_flag(self) -> None:
        metadata = MCPOAuthProtectedResourceMetadataConfig.from_dict(
            {"mcp_enable_unauthenticated_well_known_route": True}
        )

        assert metadata.mcp_enable_unauthenticated_well_known_route is True


class TestMCPOAuthProtectedResourceMetadata:
    @pytest.fixture
    def admin_config(self) -> MCPOAuthProtectedResourceMetadataAdminConfig:
        return MCPOAuthProtectedResourceMetadataAdminConfig(["header"])

    def test_build_publishes_custom_fields_with_x_prefix(
        self,
        metadata_in_dict: dict[str, Any],
        cross_application_access_in_dict: dict[str, Any],
        admin_config: MCPOAuthProtectedResourceMetadataAdminConfig,
    ) -> None:
        metadata_in_dict["mcp_enable_unauthenticated_well_known_route"] = True
        user_config = MCPOAuthProtectedResourceMetadataConfig.from_dict(metadata_in_dict)

        served = MCPOAuthProtectedResourceMetadata.build(
            user_config, admin_config
        ).to_dict_without_null_attribute()

        # Registered RFC 9728 parameters keep their standard names.
        assert served["resource"] == metadata_in_dict["resource"]
        assert served["bearer_methods_supported"] == ["header"]
        # cross_application_access is published unprefixed by design; the other
        # DataRobot addition is x_-prefixed. The pre-rename name is gone.
        assert served["cross_application_access"] == cross_application_access_in_dict
        assert served["x_mcp_enable_unauthenticated_well_known_route"] is True
        assert "xaa_metadata" not in served
        assert "x_cross_application_access" not in served
        assert "mcp_enable_unauthenticated_well_known_route" not in served

    def test_build_omits_unset_optional_fields(
        self,
        cross_application_access_in_dict: dict[str, Any],
        admin_config: MCPOAuthProtectedResourceMetadataAdminConfig,
    ) -> None:
        user_config = MCPOAuthProtectedResourceMetadataConfig.from_dict(
            {"cross_application_access": cross_application_access_in_dict}
        )

        served = MCPOAuthProtectedResourceMetadata.build(
            user_config, admin_config
        ).to_dict_without_null_attribute()

        assert served == {
            "bearer_methods_supported": ["header"],
            "cross_application_access": cross_application_access_in_dict,
        }
