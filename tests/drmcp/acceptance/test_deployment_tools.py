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

"""Acceptance tests for deployment MCP tools (list_deployments, deploy_model, deploy_custom_model)."""  # noqa: E501

import os
from pathlib import Path

import pytest

from datarobot_genai.drmcp import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session


@pytest.mark.asyncio
class TestDeploymentToolsAcceptance:
    """Acceptance tests for deployment tools via MCP session (in-process)."""

    async def test_deploy_custom_model_tool_listed(self) -> None:
        """Verify deploy_custom_model is available in the MCP server tool list."""
        async with integration_test_mcp_session() as session:
            result = await session.list_tools()
            tool_names = [t.name for t in result.tools]
            assert "deploy_custom_model" in tool_names
            assert "list_deployments" in tool_names
            assert "deploy_model" in tool_names


@pytest.mark.skipif(
    not os.getenv("ENABLE_CUSTOM_MODEL_DEPLOY"),
    reason="ENABLE_CUSTOM_MODEL_DEPLOY not set",
)
@pytest.mark.asyncio
class TestDeployCustomModelE2E:
    """E2E tests for deploy_custom_model when a DataRobot environment is available."""

    async def test_deploy_custom_model_tool_callable(self) -> None:
        """Smoke test: call deploy_custom_model with minimal params; expect auth or validation."""  # noqa: E501
        fixture_dir = str(
            Path(__file__).resolve().parent.parent
            / "unit"
            / "predictive_tools"
            / "fixtures"
            / "custom_model"
        )
        async with ete_test_mcp_session() as session:
            result = await session.call_tool(
                "deploy_custom_model",
                {
                    "model_folder": fixture_dir,
                    "name": "E2E Test Model",
                    "target_type": "binary",
                    "target_name": "target",
                },
            )
            assert result.content is not None
            assert len(result.content) > 0
