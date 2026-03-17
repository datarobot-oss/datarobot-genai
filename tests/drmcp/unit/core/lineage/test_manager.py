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
from collections.abc import Iterator
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core.feature_flags import FeatureFlag
from datarobot_genai.drmcp.core.lineage.manager import LineageManager


class TestLineageManager:
    @pytest.fixture
    def module_under_test(self) -> str:
        return "datarobot_genai.drmcp.core.lineage.manager"

    @pytest.fixture
    def mock_get_datarobot_client(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_datarobot_client") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_feature_flag_create(self) -> Iterator[Mock]:
        with patch.object(FeatureFlag, "create") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_lrs_env_var(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.LRSEnvVars") as mock_enum:
            yield mock_enum

    def test_init(
        self,
        mock_feature_flag_create: Mock,
        mock_get_datarobot_client: Mock,
        mock_lrs_env_var: Mock,
    ) -> None:
        manager = LineageManager()

        mock_get_datarobot_client.assert_called_once_with()
        mock_feature_flag_create.assert_called_once_with("ENABLE_MCP_TOOLS_GALLERY_SUPPORT")
        mock_lrs_env_var.MLOPS_DEPLOYMENT_ID.get_os_env_value.assert_called_once_with()
        mock_lrs_env_var.MLOPS_MODEL_ID.get_os_env_value.assert_called_once_with()

        assert manager.datarobot_client == mock_get_datarobot_client.return_value
        assert manager.feature_flag_enabled == mock_feature_flag_create.return_value.enabled
        assert (
            manager.custom_model_deployment_id
            == mock_lrs_env_var.MLOPS_DEPLOYMENT_ID.get_os_env_value.return_value
        )
        assert (
            manager.custom_model_version_id
            == mock_lrs_env_var.MLOPS_MODEL_ID.get_os_env_value.return_value
        )
