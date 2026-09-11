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

"""Tests for feature-flag-aware registration in ``drtools_registry``."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core import drtools_registry


def _example_tool() -> None:
    """Serve as a placeholder subject for registration tests."""


class TestRegisterDrtoolsFunctionFeatureFlag:
    @pytest.fixture
    def mock_dr_mcp_tool(self) -> MagicMock:
        with patch.object(drtools_registry, "dr_mcp_tool") as mocked:
            mocked.return_value = lambda func: func
            yield mocked

    def test_register_skips_when_feature_flag_disabled(self, mock_dr_mcp_tool: MagicMock) -> None:
        with patch.object(
            drtools_registry, "_static_account_flag_enabled", return_value=False
        ) as mock_eval:
            drtools_registry.register_drtools_function(
                _example_tool, {"feature_flag": "X", "tags": {"foo"}}
            )

        mock_eval.assert_called_once_with("X")
        mock_dr_mcp_tool.assert_not_called()

    def test_register_proceeds_when_feature_flag_enabled(self, mock_dr_mcp_tool: MagicMock) -> None:
        with patch.object(drtools_registry, "_static_account_flag_enabled", return_value=True):
            drtools_registry.register_drtools_function(
                _example_tool, {"feature_flag": "X", "tags": {"foo"}}
            )

        mock_dr_mcp_tool.assert_called_once()
        _, kwargs = mock_dr_mcp_tool.call_args
        assert "feature_flag" not in kwargs
        assert kwargs.get("tags") == {"foo"}

    def test_register_fails_closed_when_ff_check_raises(self, mock_dr_mcp_tool: MagicMock) -> None:
        # Any error from the evaluator must fail closed (tool not registered).
        with patch.object(
            drtools_registry,
            "_static_account_flag_enabled",
            side_effect=RuntimeError("DR client unavailable"),
        ):
            drtools_registry.register_drtools_function(_example_tool, {"feature_flag": "X"})

        mock_dr_mcp_tool.assert_not_called()

    def test_register_without_feature_flag_proceeds(self, mock_dr_mcp_tool: MagicMock) -> None:
        # Sanity: existing behavior (no FF) is unchanged — evaluator never consulted.
        with patch.object(drtools_registry, "_static_account_flag_enabled") as mock_eval:
            drtools_registry.register_drtools_function(_example_tool, {"tags": {"foo"}})

        mock_eval.assert_not_called()
        mock_dr_mcp_tool.assert_called_once()
