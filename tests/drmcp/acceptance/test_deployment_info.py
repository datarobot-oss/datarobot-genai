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
from pathlib import Path
from typing import Any

import pytest

from datarobot_genai.drmcp.test_utils.mcp_utils_ete import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import ANY_NONEMPTY_STRING
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY
from datarobot_genai.drmcp.test_utils.tool_base_ete import ETETestExpectations
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolBaseE2E
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolCallTestExpectations

# Matches text classification deployment features; no tricky quoting so the LLM can copy it.
INLINE_CSV_FOR_VALIDATE = "text_review,product_category\nhello world,electronics\n"


@pytest.fixture(scope="session")
def expectations_for_get_deployment_features_success(
    deployment_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="get_deployment_features",
                parameters={"deployment_id": deployment_id},
                result={"total_features": 0, "features": []},
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
                parameters={"deployment_id": deployment_id},
                result={
                    "deployment_id": "",
                    "model_type": "",
                    "target": "",
                    "target_type": "",
                    "total_features": 0,
                    "template_data": [],
                },
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
    csv_string = Path(diabetes_scoring_small_file_path).read_text()
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="validate_prediction_data",
                parameters={
                    "deployment_id": deployment_id,
                    "csv_string": csv_string,
                },
                result={"status": "valid"},
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
def expectations_for_validate_prediction_data_inline_csv(
    deployment_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="validate_prediction_data",
                parameters={
                    "deployment_id": deployment_id,
                    "csv_string": ANY_NONEMPTY_STRING,
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "valid",
            "validate",
            "deployment",
            "csv",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_validate_prediction_data_failure(
    deployment_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="validate_prediction_data",
                parameters={
                    "deployment_id": deployment_id,
                    "csv_string": "",
                },
                result="csv_string",
            ),
        ],
        llm_response_content_contains_expectations=[
            "csv_string",
            "empty",
            "cannot be empty",
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
        llm_client: Any,
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
                llm_client,
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
        llm_client: Any,
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
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
            I have a DataRobot deployment with ID '{deployment_id}' for a text classification
            model. Check whether this exact multi-line CSV is valid for scoring (preserve
            newlines; do not add a file path):

            {csv_inline}
            """
        ],
    )
    async def test_validate_prediction_data_inline_csv_success(
        self,
        llm_client: Any,
        expectations_for_validate_prediction_data_inline_csv: ETETestExpectations,
        deployment_id: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            deployment_id=deployment_id,
            csv_inline=INLINE_CSV_FOR_VALIDATE,
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = (
                frame.f_code.co_name
                if frame
                else "test_validate_prediction_data_inline_csv_success"
            )
            await self._run_test_with_expectations(
                prompt,
                expectations_for_validate_prediction_data_inline_csv,
                llm_client,
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
        llm_client: Any,
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
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
            DataRobot deployment_id is '{deployment_id}'.
            Try to validate scoring data where the inline CSV body is completely empty: pass
            csv_string as exactly an empty string (length zero, no spaces, no placeholder text).
            Do not use file paths. We expect a clear validation or argument error for empty input.
            """
        ],
    )
    async def test_validate_prediction_data_failure(
        self,
        llm_client: Any,
        expectations_for_validate_prediction_data_failure: ETETestExpectations,
        deployment_id: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(deployment_id=deployment_id)

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_validate_prediction_data_failure"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_validate_prediction_data_failure,
                llm_client,
                session,
                test_name,
            )
