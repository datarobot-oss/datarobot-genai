# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path

import pytest
from nat.data_models.config import Config
from nat.runtime.loader import load_config

import datarobot_genai.dragent.plugins.datarobot_user_mcp_xaa_client  # noqa: F401 — registers mcp_client_with_xaa_support
import datarobot_genai.dragent.plugins.okta_a2a_auth  # noqa: F401 — registers okta_cross_app_access
from datarobot_genai.dragent.plugins.datarobot_user_mcp_xaa_client import (
    MCPClientWithXAASupportConfig,
)
from datarobot_genai.dragent.plugins.okta_a2a_auth import (
    OAuth2CrossApplicationAccessAuthProviderConfig,
)


@pytest.fixture(scope="module")
def workflow_path() -> Path:
    return Path(__file__).parent / "fixtures" / "workflow_with_datarobot_user_mcp_xaa_client.yaml"


@pytest.fixture
def nat_config(workflow_path: Path) -> Config:
    return load_config(workflow_path)


@pytest.fixture
def auth_provider_id_in_nat_workflow() -> str:
    return "okta_auth"


@pytest.fixture
def mcp_func_group_id_in_nat_workflow() -> str:
    return "mcp"


class TestConfigParsedFromYaml:
    def test_auth_provider_config(
        self, nat_config: Config, auth_provider_id_in_nat_workflow: str
    ) -> None:
        assert isinstance(
            nat_config.authentication[auth_provider_id_in_nat_workflow],
            OAuth2CrossApplicationAccessAuthProviderConfig,
        )

    def test_mcp_func_group_config(
        self, nat_config: Config, mcp_func_group_id_in_nat_workflow: str
    ) -> None:
        assert isinstance(
            nat_config.function_groups[mcp_func_group_id_in_nat_workflow],
            MCPClientWithXAASupportConfig,
        )
