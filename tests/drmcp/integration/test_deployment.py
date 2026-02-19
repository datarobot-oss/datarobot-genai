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

from pathlib import Path

import pytest
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session


def _custom_model_fixture_dir() -> str:
    return str(
        Path(__file__).resolve().parent.parent
        / "unit"
        / "predictive_tools"
        / "fixtures"
        / "custom_model"
    )


@pytest.mark.asyncio
class TestMCPDeploymentIntegration:
    """Integration tests for MCP deployment tools."""

    async def test_deploy_custom_model_tool_registered(self) -> None:
        """Verify deploy_custom_model is registered and callable via MCP session."""
        async with integration_test_mcp_session() as session:
            result = await session.list_tools()
            tool_names = [t.name for t in result.tools]
            assert "deploy_custom_model" in tool_names

    async def test_deploy_custom_model_call_validation_error(self) -> None:
        """Call deploy_custom_model with fixture folder; expect error (validation or deployment)."""
        folder = _custom_model_fixture_dir()
        async with integration_test_mcp_session() as session:
            result = await session.call_tool(
                "deploy_custom_model",
                {
                    "model_folder": folder,
                    "name": "Test",
                    "target_type": "binary",
                    "target_name": "target",
                },
            )
            assert result.isError or len(result.content) == 0
            if result.content:
                content = result.content[0]
                assert isinstance(content, TextContent)
                text = content.text.lower()
                assert (
                    "model file" in text
                    or "model_file_path" in text
                    or "token" in text
                    or "custom model did not start" in text
                    or "job did not complete" in text
                    or "error" in text
                )
