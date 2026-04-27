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
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core.feature_flags import FeatureFlag


class TestFeatureFlags:
    @pytest.fixture
    def module_under_test(self) -> str:
        return "datarobot_genai.drmcp.core.feature_flags"

    @pytest.fixture
    def mock_get_datarobot_client(self, module_under_test: str) -> Iterator[Mock]:
        with patch(
            f"{module_under_test}.setup_and_return_dr_api_client_with_static_config_in_container"
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_feature_flag_create(self) -> Iterator[Mock]:
        with patch.object(FeatureFlag, "create") as mock_func:
            yield mock_func

    @pytest.mark.parametrize("feature_flag_value", [True, False], ids=str)
    def test_create(self, mock_get_datarobot_client: Mock, feature_flag_value: bool) -> None:
        mock_datarobot_client = mock_get_datarobot_client.return_value
        expected_feature_flag_name = Mock()
        mock_datarobot_client.post.return_value.json.return_value = {
            "entitlements": [{"name": expected_feature_flag_name, "value": feature_flag_value}]
        }

        output = FeatureFlag.create(expected_feature_flag_name)

        mock_datarobot_client.post.assert_called_once_with(
            "entitlements/evaluate/",
            json={"entitlements": [{"name": expected_feature_flag_name}]},
        )
        assert output == FeatureFlag(
            name=expected_feature_flag_name,
            enabled=feature_flag_value,
        )

    @pytest.mark.parametrize(
        "feature_flag_api_response",
        [{}, {"entitlements": []}, {"entitlements": [{}]}],
    )
    def test_fallback_to_feature_flag_enablement_false(
        self,
        mock_get_datarobot_client: Mock,
        feature_flag_api_response: dict[str, Any],
    ) -> None:
        mock_datarobot_client = mock_get_datarobot_client.return_value
        mock_datarobot_client.post.return_value.json.return_value = feature_flag_api_response

        expected_feature_flag_name = Mock()
        output = FeatureFlag.create(expected_feature_flag_name)

        mock_datarobot_client.post.assert_called_once_with(
            "entitlements/evaluate/",
            json={"entitlements": [{"name": expected_feature_flag_name}]},
        )
        assert output == FeatureFlag(name=expected_feature_flag_name, enabled=False)

    def test_is_mcp_tools_gallery_support_enabled(self, mock_feature_flag_create: Mock) -> None:
        output = FeatureFlag.is_mcp_tools_gallery_support_enabled()

        mock_feature_flag_create.assert_called_once_with("ENABLE_MCP_TOOLS_GALLERY_SUPPORT")
        assert output == mock_feature_flag_create.return_value.enabled
