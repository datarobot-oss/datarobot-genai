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
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import split_list_setting

TRUSTED_ISSUER = "https://foo/bar/issuer"
EXCHANGE_AUDIENCE = "https://foo/bar/token_exchange_audience"
TOKEN_URL = "https://foo/bar/token"
TOKEN_AUDIENCE = "https://foo/bar/token_request_audience"
RESOURCE_URL = "https://foo/bar/mcp_resource_server"


@pytest.fixture
def xaa_settings() -> dict[str, Any]:
    return {
        "trusted_issuer": TRUSTED_ISSUER,
        "exchange_audience": EXCHANGE_AUDIENCE,
        "token_url": TOKEN_URL,
        "token_audience": TOKEN_AUDIENCE,
        "scopes": "scope",
    }


@pytest.fixture
def cross_application_access_in_dict() -> dict[str, Any]:
    return {
        "token_endpoint_auth_method": DEFAULT_TOKEN_ENDPOINT_AUTH_METHOD,
        "token_exchange": {
            "trusted_issuer": TRUSTED_ISSUER,
            "audience": EXCHANGE_AUDIENCE,
        },
        "token_request": {
            "token_url": TOKEN_URL,
            "audience": TOKEN_AUDIENCE,
            "scopes": ["scope"],
        },
    }


@dataclass
class DummyDataClassInheritingBaseDataClass(BaseDataClass):
    attribute: int
    nullable_attribute: int | None


class TestBaseDataClass:
    def test_to_dict_without_null_attribute(self) -> None:
        dataclass_object = DummyDataClassInheritingBaseDataClass(1, None)
        assert dataclass_object.to_dict_without_null_attribute() == {"attribute": 1}


class TestSplitListSetting:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            ("", None),
            ("   ", None),
            (",", None),
            ("one", ["one"]),
            (" one , two ,", ["one", "two"]),
        ],
    )
    def test_split(self, value: str | None, expected: list[str] | None) -> None:
        assert split_list_setting(value) == expected


class TestCrossApplicationAccessMetadata:
    def test_from_settings(self, xaa_settings: dict[str, Any]) -> None:
        metadata = CrossApplicationAccessMetadata.from_settings(**xaa_settings)

        assert isinstance(metadata, CrossApplicationAccessMetadata)
        assert metadata.token_endpoint_auth_method == DEFAULT_TOKEN_ENDPOINT_AUTH_METHOD
        token_exchange_params = metadata.token_exchange
        assert isinstance(token_exchange_params, XAATokenExchangeParams)
        assert token_exchange_params.trusted_issuer == TRUSTED_ISSUER
        assert token_exchange_params.audience == EXCHANGE_AUDIENCE
        token_request_params = metadata.token_request
        assert isinstance(token_request_params, XAATokenRequestParams)
        assert token_request_params.token_url == TOKEN_URL
        assert token_request_params.audience == TOKEN_AUDIENCE
        assert token_request_params.scopes == ["scope"]

    def test_from_settings_without_token_request_audience(
        self, xaa_settings: dict[str, Any]
    ) -> None:
        xaa_settings.pop("token_audience")

        metadata = CrossApplicationAccessMetadata.from_settings(**xaa_settings)

        assert metadata is not None
        assert metadata.token_request.audience is None

    def test_from_settings_without_any_setting(self) -> None:
        assert CrossApplicationAccessMetadata.from_settings() is None

    #: Maps each required ``from_settings`` argument to the variable that sets it.
    REQUIRED_ENV_NAMES = {
        "trusted_issuer": "MCP_XAA_TRUSTED_ISSUER",
        "exchange_audience": "MCP_XAA_EXCHANGE_AUDIENCE",
        "token_url": "MCP_XAA_TOKEN_URL",
        "scopes": "MCP_XAA_SCOPES",
    }

    @pytest.mark.parametrize(
        "omitted",
        ["trusted_issuer", "exchange_audience", "token_url", "scopes"],
    )
    def test_from_settings_drops_a_partial_block_naming_what_is_missing(
        self,
        xaa_settings: dict[str, Any],
        omitted: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The deploy tooling rejects this up front; here it must not 500.

        The warning names exactly the variable that is missing — the fix is
        setting that one variable, so the message should say which.
        """
        xaa_settings.pop(omitted)

        assert CrossApplicationAccessMetadata.from_settings(**xaa_settings) is None
        assert "Incomplete Cross-Application Access settings" in caplog.text
        assert self.REQUIRED_ENV_NAMES[omitted] in caplog.text
        for present, env_name in self.REQUIRED_ENV_NAMES.items():
            if present != omitted:
                assert env_name not in caplog.text

    def test_every_missing_setting_is_named_at_once(
        self, xaa_settings: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        """One warning lists the whole gap, not one variable per restart."""
        xaa_settings.pop("token_url")
        xaa_settings.pop("scopes")

        assert CrossApplicationAccessMetadata.from_settings(**xaa_settings) is None
        assert "MCP_XAA_TOKEN_URL" in caplog.text
        assert "MCP_XAA_SCOPES" in caplog.text
        assert "MCP_XAA_TRUSTED_ISSUER" not in caplog.text

    def test_token_endpoint_auth_method_defaults_when_omitted(
        self, xaa_settings: dict[str, Any]
    ) -> None:
        metadata = CrossApplicationAccessMetadata.from_settings(**xaa_settings)

        assert metadata is not None
        assert metadata.token_endpoint_auth_method == DEFAULT_TOKEN_ENDPOINT_AUTH_METHOD
        assert DEFAULT_TOKEN_ENDPOINT_AUTH_METHOD == "private_key_jwt"

    def test_token_endpoint_auth_method_is_overridable(self, xaa_settings: dict[str, Any]) -> None:
        metadata = CrossApplicationAccessMetadata.from_settings(
            token_endpoint_auth_method="client_secret_jwt", **xaa_settings
        )

        assert metadata is not None
        assert metadata.token_endpoint_auth_method == "client_secret_jwt"


class TestMCPOAuthProtectedResourceMetadataConfig:
    def test_from_settings(
        self, xaa_settings: dict[str, Any], cross_application_access_in_dict: dict[str, Any]
    ) -> None:
        metadata = MCPOAuthProtectedResourceMetadataConfig.from_settings(
            resource=RESOURCE_URL,
            authorization_servers="https://as1,https://as2",
            scopes_supported=["scope"],
            xaa_trusted_issuer=xaa_settings["trusted_issuer"],
            xaa_exchange_audience=xaa_settings["exchange_audience"],
            xaa_token_url=xaa_settings["token_url"],
            xaa_token_audience=xaa_settings["token_audience"],
            xaa_scopes=xaa_settings["scopes"],
        )

        assert metadata.resource == RESOURCE_URL
        assert metadata.authorization_servers == ["https://as1", "https://as2"]
        assert metadata.scopes_supported == ["scope"]
        assert metadata.to_dict_without_null_attribute()["cross_application_access"] == (
            cross_application_access_in_dict
        )

    def test_from_settings_with_only_cross_application_access(
        self, xaa_settings: dict[str, Any], cross_application_access_in_dict: dict[str, Any]
    ) -> None:
        """resource/authorization_servers/scopes_supported have no logic behind them yet."""
        metadata = MCPOAuthProtectedResourceMetadataConfig.from_settings(
            xaa_trusted_issuer=xaa_settings["trusted_issuer"],
            xaa_exchange_audience=xaa_settings["exchange_audience"],
            xaa_token_url=xaa_settings["token_url"],
            xaa_token_audience=xaa_settings["token_audience"],
            xaa_scopes=xaa_settings["scopes"],
        )

        assert metadata.resource is None
        assert metadata.authorization_servers is None
        assert metadata.scopes_supported is None
        assert metadata.to_dict_without_null_attribute() == {
            "cross_application_access": cross_application_access_in_dict
        }

    def test_scopes_supported_blank_entries_are_dropped(self) -> None:
        metadata = MCPOAuthProtectedResourceMetadataConfig.from_settings(
            scopes_supported=["mcp:tools:read", "  ", ""]
        )

        assert metadata.scopes_supported == ["mcp:tools:read"]

    def test_no_scopes_publishes_no_scopes_supported(self) -> None:
        """An empty list is an unset field, not an empty array in the document."""
        metadata = MCPOAuthProtectedResourceMetadataConfig.from_settings(scopes_supported=[])

        assert metadata.scopes_supported is None

    def test_from_settings_with_nothing_set(self) -> None:
        metadata = MCPOAuthProtectedResourceMetadataConfig.from_settings()

        assert metadata.cross_application_access is None
        assert metadata.to_dict_without_null_attribute() == {}
        assert metadata.is_empty() is True

    def test_blank_settings_count_as_unset(self) -> None:
        metadata = MCPOAuthProtectedResourceMetadataConfig.from_settings(
            resource="", authorization_servers="  "
        )

        assert metadata.is_empty() is True


class TestMCPOAuthProtectedResourceMetadata:
    @pytest.fixture
    def admin_config(self) -> MCPOAuthProtectedResourceMetadataAdminConfig:
        return MCPOAuthProtectedResourceMetadataAdminConfig(["header"])

    def test_build_publishes_registered_and_datarobot_fields(
        self,
        xaa_settings: dict[str, Any],
        cross_application_access_in_dict: dict[str, Any],
        admin_config: MCPOAuthProtectedResourceMetadataAdminConfig,
    ) -> None:
        user_config = MCPOAuthProtectedResourceMetadataConfig.from_settings(
            resource=RESOURCE_URL,
            xaa_trusted_issuer=xaa_settings["trusted_issuer"],
            xaa_exchange_audience=xaa_settings["exchange_audience"],
            xaa_token_url=xaa_settings["token_url"],
            xaa_token_audience=xaa_settings["token_audience"],
            xaa_scopes=xaa_settings["scopes"],
        )

        served = MCPOAuthProtectedResourceMetadata.build(
            user_config, admin_config
        ).to_dict_without_null_attribute()

        # Registered RFC 9728 parameters keep their standard names.
        assert served["resource"] == RESOURCE_URL
        assert served["bearer_methods_supported"] == ["header"]
        # cross_application_access is published unprefixed by design, matching
        # the agent-side config block. The pre-rename name is gone.
        assert served["cross_application_access"] == cross_application_access_in_dict
        assert "xaa_metadata" not in served
        assert "x_cross_application_access" not in served

    def test_build_publishes_no_server_configuration_fields(
        self,
        admin_config: MCPOAuthProtectedResourceMetadataAdminConfig,
    ) -> None:
        """The document describes the resource, not how the server is routed."""
        served = MCPOAuthProtectedResourceMetadata.build(
            MCPOAuthProtectedResourceMetadataConfig(resource=RESOURCE_URL), admin_config
        ).to_dict_without_null_attribute()

        assert not [key for key in served if key.startswith("x_")]
        assert "mcp_enable_unauthenticated_well_known_route" not in served

    def test_build_omits_unset_optional_fields(
        self,
        xaa_settings: dict[str, Any],
        cross_application_access_in_dict: dict[str, Any],
        admin_config: MCPOAuthProtectedResourceMetadataAdminConfig,
    ) -> None:
        user_config = MCPOAuthProtectedResourceMetadataConfig.from_settings(
            xaa_trusted_issuer=xaa_settings["trusted_issuer"],
            xaa_exchange_audience=xaa_settings["exchange_audience"],
            xaa_token_url=xaa_settings["token_url"],
            xaa_token_audience=xaa_settings["token_audience"],
            xaa_scopes=xaa_settings["scopes"],
        )

        served = MCPOAuthProtectedResourceMetadata.build(
            user_config, admin_config
        ).to_dict_without_null_attribute()

        assert served == {
            "bearer_methods_supported": ["header"],
            "cross_application_access": cross_application_access_in_dict,
        }
