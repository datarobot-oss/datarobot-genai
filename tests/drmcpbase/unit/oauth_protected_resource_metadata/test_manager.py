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
import os
from collections.abc import Iterator
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    MCPOAuthProtectedResourceMetadata,
)
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    MCPOAuthProtectedResourceMetadataAdminConfig,
)
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.entities import (
    MCPOAuthProtectedResourceMetadataConfig,
)
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.manager import ContainerEnvVar
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.manager import (
    MCPOAuthProtectedResourceMetadataManager,
)
from datarobot_genai.drmcpbase.oauth_protected_resource_metadata.manager import (
    SupportedMethodsToSendBearerToken,
)


class TestContainerEnvVar:
    @pytest.fixture
    def mock_os_getenv(self) -> Iterator[Mock]:
        with patch.object(os, "getenv") as mock_func:
            yield mock_func

    @pytest.mark.parametrize("env_var_enum", ContainerEnvVar, ids=str)
    def test_get_env_var_value(self, env_var_enum: ContainerEnvVar, mock_os_getenv: Mock) -> None:
        env_var_enum.get_env_var_value()

        mock_os_getenv.assert_called_once_with(env_var_enum.name)


class TestSupportedMethodsToSendBearerToken:
    @pytest.mark.parametrize("supported_method_enum", SupportedMethodsToSendBearerToken, ids=str)
    def test_get_name_in_lower_case(
        self,
        supported_method_enum: SupportedMethodsToSendBearerToken,
    ) -> None:
        enum_and_names = {
            SupportedMethodsToSendBearerToken.HEADER: "header",
        }
        assert (
            supported_method_enum.get_name_in_lower_case() == enum_and_names[supported_method_enum]
        )

    def test_get_complete_list_of_supported_methods(self) -> None:
        outputs = SupportedMethodsToSendBearerToken.get_complete_list_of_supported_methods()
        assert outputs == ["header"]


class TestMCPOAuthProtectedResourceMetadataManager:
    @pytest.fixture
    def mock_json_loads(self) -> Iterator[Mock]:
        with patch.object(json, "loads") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_load_mcp_oauth_protected_resource_metadata_from_env_var(self) -> Iterator[Mock]:
        with patch.object(
            ContainerEnvVar.MCP_OAUTH_PROTECTED_RESOURCE_METADATA,
            "get_env_var_value",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_mcp_oauth_protected_resource_metadata_user_config_from_json(self) -> Iterator[Mock]:
        with patch.object(MCPOAuthProtectedResourceMetadataConfig, "from_json") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_build_mcp_oauth_protected_resource_metadata(self) -> Iterator[Mock]:
        with patch.object(MCPOAuthProtectedResourceMetadata, "build") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_load_config(self) -> Iterator[Mock]:
        with patch.object(MCPOAuthProtectedResourceMetadataManager, "load_config") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_admin_config(self) -> Iterator[Mock]:
        with patch.object(
            MCPOAuthProtectedResourceMetadataManager, "get_admin_config"
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_protected_resource_metadata(self) -> Iterator[Mock]:
        with patch.object(
            MCPOAuthProtectedResourceMetadataManager,
            "get_protected_resource_metadata",
        ) as mock_func:
            yield mock_func

    def test_load_config(
        self,
        mock_json_loads: Mock,
        mock_load_mcp_oauth_protected_resource_metadata_from_env_var: Mock,
        mock_mcp_oauth_protected_resource_metadata_user_config_from_json: Mock,
    ) -> None:
        manager = MCPOAuthProtectedResourceMetadataManager()
        output = manager.load_config()

        mock_load_mcp_oauth_protected_resource_metadata_from_env_var.assert_called_once_with()
        mock_metadata_json_string = (
            mock_load_mcp_oauth_protected_resource_metadata_from_env_var.return_value
        )
        mock_json_loads.assert_called_once_with(mock_metadata_json_string)
        mock_mcp_oauth_protected_resource_metadata_user_config_from_json.assert_called_once_with(
            mock_json_loads.return_value
        )
        assert (
            output == mock_mcp_oauth_protected_resource_metadata_user_config_from_json.return_value
        )

    def test_get_admin_config(self) -> None:
        output = MCPOAuthProtectedResourceMetadataManager().get_admin_config()

        assert isinstance(output, MCPOAuthProtectedResourceMetadataAdminConfig)
        assert output.bearer_methods_supported == ["header"]

    def test_get_protected_resource_metadata(
        self,
        mock_build_mcp_oauth_protected_resource_metadata: Mock,
        mock_get_admin_config: Mock,
        mock_load_config: Mock,
    ) -> None:
        manager = MCPOAuthProtectedResourceMetadataManager()
        output = manager.get_protected_resource_metadata()

        mock_load_config.assert_called_once_with()
        mock_get_admin_config.assert_called_once_with()
        mock_build_mcp_oauth_protected_resource_metadata.assert_called_once_with(
            mock_load_config.return_value,
            mock_get_admin_config.return_value,
        )
        assert output == mock_build_mcp_oauth_protected_resource_metadata.return_value

    def test_get_protected_resource_metadata_return_none(
        self,
        mock_build_mcp_oauth_protected_resource_metadata: Mock,
        mock_get_admin_config: Mock,
        mock_load_config: Mock,
    ) -> None:
        mock_load_config.return_value = None

        manager = MCPOAuthProtectedResourceMetadataManager()
        output = manager.get_protected_resource_metadata()

        mock_load_config.assert_called_once_with()
        mock_get_admin_config.assert_not_called()
        mock_build_mcp_oauth_protected_resource_metadata.assert_not_called()
        assert output is None

    def test_get_protected_resource_metadata_api_response(
        self,
        mock_get_protected_resource_metadata: Mock,
    ) -> None:
        manager = MCPOAuthProtectedResourceMetadataManager()
        output = manager.get_protected_resource_metadata_api_response()

        mock_metadata = mock_get_protected_resource_metadata.return_value
        mock_metadata.to_json_without_null_attribute.assert_called_once_with()
        assert output == mock_metadata.to_json_without_null_attribute.return_value

    def test_get_protected_resource_metadata_api_response_return_none(
        self,
        mock_get_protected_resource_metadata: Mock,
    ) -> None:
        mock_get_protected_resource_metadata.return_value = None

        manager = MCPOAuthProtectedResourceMetadataManager()
        output = manager.get_protected_resource_metadata_api_response()

        assert output is None
