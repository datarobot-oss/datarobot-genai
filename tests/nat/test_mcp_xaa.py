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
from nat.runtime.loader import load_config

import datarobot_genai.dragent.plugins.mcp_xaa_auth  # noqa: F401 — registers mcp_xaa_auth_provider
from datarobot_genai.dragent.plugins.mcp_xaa_auth import MCPXAAAuthProviderConfig


@pytest.fixture(autouse=True)
def set_datarobot_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a DataRobot API token before the config is parsed.

    ``DataRobotAPIKeyAuthProviderConfig`` reads ``DATAROBOT_API_TOKEN`` at
    parse time (``default_factory=_get_default_api_token``), so the env var
    must be present **before** ``load_config()`` runs.
    """
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "integration-test-token")


@pytest.fixture(scope="module")
def workflow_path() -> Path:
    return Path(__file__).parent / "fixtures" / "workflow_with_mcp_xaa.yaml"


@pytest.fixture
def nat_config(workflow_path: Path):
    return load_config(workflow_path)


class TestConfigParsedFromYaml:
    """Validate that all function_groups and auth providers
    are parsed from workflow_* yaml into the correct config types with the expected field values.
    """

    def test_mcp_xaa_auth_provider(self, nat_config):
        assert isinstance(
            nat_config.authentication["okta_auth"],
            MCPXAAAuthProviderConfig,
        )
