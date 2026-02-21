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

import inspect
import os
from pathlib import Path
from typing import Any

import pytest

from datarobot_genai.drmcp import ETETestExpectations
from datarobot_genai.drmcp import ToolBaseE2E
from datarobot_genai.drmcp import ToolCallTestExpectations
from datarobot_genai.drmcp import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY


def _custom_model_fixture_dir() -> str:
    return str(
        Path(__file__).resolve().parent.parent
        / "unit"
        / "predictive_tools"
        / "fixtures"
        / "custom_model"
    )


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
class TestDeployCustomModelE2E(ToolBaseE2E):
    """E2E tests for deploy_custom_model when a DataRobot environment is available."""

    async def test_deploy_custom_model_tool_callable(self, llm_client: Any) -> None:
        """Smoke test: LLM is prompted to deploy custom model; expect tool use and response."""  # noqa: E501
        model_folder = _custom_model_fixture_dir()
        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="deploy_custom_model",
                    parameters={
                        "model_folder": model_folder,
                        "name": "E2E Test Model",
                        "target_type": "binary",
                        "target_name": "target",
                    },
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["deploy", "model"],
        )
        prompt = (
            f"I have a custom model in folder '{model_folder}'. "
            "Please deploy it to DataRobot with name 'E2E Test Model', "
            "target type binary, and target name 'target'."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_deploy_custom_model_tool_callable"
            await self._run_test_with_expectations(
                prompt,
                expectations,
                llm_client,
                session,
                test_name,
            )
