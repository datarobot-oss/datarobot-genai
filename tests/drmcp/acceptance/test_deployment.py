# Copyright 2025 DataRobot, Inc.
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

import inspect
from typing import Any

import pytest

from datarobot_genai.drmcp.test_utils.mcp_utils_ete import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY
from datarobot_genai.drmcp.test_utils.tool_base_ete import ETETestExpectations
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolBaseE2E
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolCallTestExpectations


@pytest.fixture(scope="session")
def expectations_for_list_deployments_success() -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="list_deployments",
                parameters={},
                result={"deployments": {}},
            ),
        ],
        llm_response_content_contains_expectations=[
            "list of deployments",
            "deployments",
            "deployment",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_get_model_info_from_deployment_success(
    deployment_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="get_model_info_from_deployment",
                parameters={"deployment_id": deployment_id},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=["model info", "model"],
    )


@pytest.fixture(scope="session")
def expectations_for_get_model_info_from_deployment_failure(
    nonexistent_deployment_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="get_model_info_from_deployment",
                parameters={"deployment_id": nonexistent_deployment_id},
                result=(
                    "Error in "
                    "get_model_info_from_deployment: ClientError: 404 client error: "
                    "{'message': 'Not Found'}"
                ),
            ),
        ],
        llm_response_content_contains_expectations=[
            "Deployment with ID",
            "Deployment not found",
            "not found",
            "does not exist",
            "unable to",
            nonexistent_deployment_id,
        ],
    )


@pytest.mark.asyncio
class TestDeploymentE2E(ToolBaseE2E):
    """End-to-end tests for deployment-related functionality."""

    @pytest.mark.parametrize(
        "prompt",
        [
            """
        I'm working on a machine learning project and I need to list all the deployments I
        have access to. Can you help me list all the deployments?
        """
        ],
    )
    async def test_list_deployments_success(
        self,
        llm_client: Any,
        expectations_for_list_deployments_success: ETETestExpectations,
        prompt: str,
    ) -> None:
        async with ete_test_mcp_session() as session:
            await self._run_test_with_expectations(
                prompt,
                expectations_for_list_deployments_success,
                llm_client,
                session,
                (
                    inspect.currentframe().f_code.co_name  # type: ignore[union-attr]
                    if inspect.currentframe()
                    else "test_list_deployments_success"
                ),
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I'm working on a machine learning project with deployment ID '{deployment_id}'.
        I need to get the model info from the deployment. Can you help me get the model
        info from the deployment?
        """
        ],
    )
    async def test_get_model_info_from_deployment_success(
        self,
        llm_client: Any,
        expectations_for_get_model_info_from_deployment_success: ETETestExpectations,
        deployment_id: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(deployment_id=deployment_id)

        async with ete_test_mcp_session() as session:
            await self._run_test_with_expectations(
                prompt,
                expectations_for_get_model_info_from_deployment_success,
                llm_client,
                session,
                (
                    inspect.currentframe().f_code.co_name  # type: ignore[union-attr]
                    if inspect.currentframe()
                    else "test_get_model_info_from_deployment_success"
                ),
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I'm working on a machine learning project with deployment ID '{deployment_id}'.
        I need to get the model info from the deployment. Can you help me get the model
        info from the deployment?
        """
        ],
    )
    async def test_get_model_info_from_deployment_failure(
        self,
        llm_client: Any,
        expectations_for_get_model_info_from_deployment_failure: ETETestExpectations,
        nonexistent_deployment_id: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(deployment_id=nonexistent_deployment_id)

        async with ete_test_mcp_session() as session:
            await self._run_test_with_expectations(
                prompt,
                expectations_for_get_model_info_from_deployment_failure,
                llm_client,
                session,
                (
                    inspect.currentframe().f_code.co_name  # type: ignore[union-attr]
                    if inspect.currentframe()
                    else "test_get_model_info_from_deployment_failure"
                ),
            )
