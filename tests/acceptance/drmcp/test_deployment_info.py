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
from datarobot_genai.drmcp.test_utils.tool_base_ete import ETETestExpectations
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolBaseE2E
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolCallTestExpectations


@pytest.fixture(scope="session")
def expectations_for_get_deployment_features_success(
    deployment_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="get_deployment_features",
                parameters={"deployment_id": deployment_id},
                result="total_features",
            ),
        ],
        llm_response_content_contains_expectations=[
            "features",
            "feature types",
            "importance scores",
            "required input features",
            "feature name",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_generate_prediction_data_template_success(
    deployment_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="generate_prediction_data_template",
                parameters={
                    "deployment_id": deployment_id,
                    "n_rows": 5,
                },
                result=f"# Prediction Data Template for Deployment: {deployment_id}",
            ),
        ],
        llm_response_content_contains_expectations=[
            "template",
            "CSV",
            "sample data",
            "generated",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_validate_prediction_data_success(
    deployment_id: str,
    diabetes_scoring_small_file_path: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="validate_prediction_data",
                parameters={
                    "deployment_id": deployment_id,
                    "file_path": str(diabetes_scoring_small_file_path),
                },
                result='"status": "valid"',
            ),
        ],
        llm_response_content_contains_expectations=[
            "valid",
            "suitable",
            "can be used",
            "ready for predictions",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_validate_prediction_data_failure(
    deployment_id: str,
    nonexistent_file_path: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="validate_prediction_data",
                parameters={
                    "deployment_id": deployment_id,
                    "file_path": nonexistent_file_path,
                },
                result=f"[Errno 2] No such file or directory: '{nonexistent_file_path}'",
            ),
        ],
        llm_response_content_contains_expectations=[
            "file does not exist",
            "cannot find the file",
            "not found",
            nonexistent_file_path,
        ],
    )


@pytest.mark.asyncio
class TestDeploymentInfoE2E(ToolBaseE2E):
    """End-to-end tests for deployment info functionality."""

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
            I have a DataRobot deployment with ID '{deployment_id}' and I need to understand
            what features it requires for making predictions.
            Can you help me get deployment features?
            """
        ],
    )
    async def test_get_deployment_features_success(
        self,
        openai_llm_client: Any,
        expectations_for_get_deployment_features_success: ETETestExpectations,
        deployment_id: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(deployment_id=deployment_id)

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_get_deployment_features_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_get_deployment_features_success,
                openai_llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
            I have a DataRobot deployment with ID '{deployment_id}' and I need to create a
            template CSV file for making predictions. Can you help me generate a template with
            5 rows of sample data that matches the deployment's requirements?
            """
        ],
    )
    async def test_generate_prediction_data_template_success(
        self,
        openai_llm_client: Any,
        expectations_for_generate_prediction_data_template_success: ETETestExpectations,
        deployment_id: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(deployment_id=deployment_id)

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = (
                frame.f_code.co_name if frame else "test_generate_prediction_data_template_success"
            )
            await self._run_test_with_expectations(
                prompt,
                expectations_for_generate_prediction_data_template_success,
                openai_llm_client,
                session,
                test_name,
            )

    @pytest.mark.skip(
        reason=(
            "Skipping this test for now until we have a way to validate score file for the "
            "classification project to replace diabetes_scoring_small_file_path"
        )
    )
    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
            I have a DataRobot deployment with ID '{deployment_id}' and a CSV file at
            '{file_path}'. Can you help me validate if this file is suitable for making
            predictions with this deployment?
            """
        ],
    )
    async def test_validate_prediction_data_success(
        self,
        openai_llm_client: Any,
        expectations_for_validate_prediction_data_success: ETETestExpectations,
        deployment_id: str,
        diabetes_scoring_small_file_path: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            deployment_id=deployment_id,
            file_path=diabetes_scoring_small_file_path,
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_validate_prediction_data_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_validate_prediction_data_success,
                openai_llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
            I have a DataRobot deployment with ID '{deployment_id}' and I need to validate a
            CSV file at '{file_path}'. Can you check if this file is suitable for making
            predictions?
            """
        ],
    )
    async def test_validate_prediction_data_failure(
        self,
        openai_llm_client: Any,
        expectations_for_validate_prediction_data_failure: ETETestExpectations,
        deployment_id: str,
        nonexistent_file_path: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            deployment_id=deployment_id,
            file_path=nonexistent_file_path,
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_validate_prediction_data_failure"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_validate_prediction_data_failure,
                openai_llm_client,
                session,
                test_name,
            )
