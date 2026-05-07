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
from datarobot_genai.drmcp.test_utils.tool_base_ete import ANY_NONEMPTY_STRING
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY
from datarobot_genai.drmcp.test_utils.tool_base_ete import ETETestExpectations
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolBaseE2E
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolCallTestExpectations

INLINE_CSV_DATASET = (
    "text_review,product_category\n"
    '"This product has incredible build quality and exceeded all my expectations",electronics\n'
    '"The software interface is clean and very easy to navigate",software'
)

# JSON rows equivalent to a small scoring batch (no file paths).
INLINE_JSON_DATASET = (
    '[{"text_review":"Compact review for testing","product_category":"electronics"},'
    '{"text_review":"Second row","product_category":"software"}]'
)


@pytest.fixture(scope="session")
def expectations_for_predict_by_ai_catalog_rt_success(
    deployment_id: str, classification_predict_dataset: Any
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="predict_by_ai_catalog_rt",
                parameters={
                    "deployment_id": deployment_id,
                    "dataset_id": classification_predict_dataset["dataset_id"],
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "prediction",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_predict_realtime_dataset_string_success(
    deployment_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="predict_realtime",
                parameters={
                    "deployment_id": deployment_id,
                    "dataset": INLINE_CSV_DATASET,
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "prediction",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_get_deployment_info_success(deployment_id: str) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="get_deployment_info",
                parameters={"deployment_id": deployment_id},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "feature",
            "deployment",
            "target",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_get_deployment_features_success(deployment_id: str) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="get_deployment_features",
                parameters={"deployment_id": deployment_id},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "feature",
            "deployment",
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
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "template",
            "column",
            "feature",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_validate_prediction_data_success(
    deployment_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="validate_prediction_data",
                parameters={
                    "deployment_id": deployment_id,
                    "csv_string": INLINE_CSV_DATASET,
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "validation",
            "deployment",
            "column",
            "row",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_predict_realtime_json_dataset_success(
    deployment_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="predict_realtime",
                parameters={
                    "deployment_id": deployment_id,
                    "dataset": ANY_NONEMPTY_STRING,
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "prediction",
        ],
    )


@pytest.mark.asyncio
class TestPredictRealtimeE2E(ToolBaseE2E):
    """End-to-end tests for realtime prediction functionality."""

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I have a DataRobot deployment with ID '{deployment_id}'.
        Please run realtime predictions using the AI Catalog dataset with ID '{dataset_id}'.
        """
        ],
    )
    async def test_predict_by_ai_catalog_rt_success(
        self,
        llm_client: Any,
        expectations_for_predict_by_ai_catalog_rt_success: ETETestExpectations,
        deployment_id: str,
        classification_predict_dataset: Any,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            deployment_id=deployment_id,
            dataset_id=classification_predict_dataset["dataset_id"],
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_predict_by_ai_catalog_rt_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_predict_by_ai_catalog_rt_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I have a DataRobot deployment with ID '{deployment_id}'.
        Please run realtime predictions using this inline CSV data in dataset parameter:
        {dataset}
        """
        ],
    )
    async def test_predict_realtime_dataset_string_success(
        self,
        llm_client: Any,
        expectations_for_predict_realtime_dataset_string_success: ETETestExpectations,
        deployment_id: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            deployment_id=deployment_id,
            dataset=INLINE_CSV_DATASET,
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = (
                frame.f_code.co_name if frame else "test_predict_realtime_dataset_string_success"
            )
            await self._run_test_with_expectations(
                prompt,
                expectations_for_predict_realtime_dataset_string_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I have a DataRobot deployment with ID '{deployment_id}'.
        Run realtime scoring using the dataset argument only as an inline JSON array of row objects
        (not a file path). Use this exact JSON string as dataset (copy verbatim):

        {dataset}
        """
        ],
    )
    async def test_predict_realtime_json_dataset_success(
        self,
        llm_client: Any,
        expectations_for_predict_realtime_json_dataset_success: ETETestExpectations,
        deployment_id: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            deployment_id=deployment_id,
            dataset=INLINE_JSON_DATASET,
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = (
                frame.f_code.co_name if frame else "test_predict_realtime_json_dataset_success"
            )
            await self._run_test_with_expectations(
                prompt,
                expectations_for_predict_realtime_json_dataset_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        For DataRobot deployment '{deployment_id}', look up what that deployment needs for
        scoring: the prediction target and the input features. Summarize the target and roughly
        how many features are required.
        """
        ],
    )
    async def test_get_deployment_info_success(
        self,
        llm_client: Any,
        expectations_for_get_deployment_info_success: ETETestExpectations,
        deployment_id: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(deployment_id=deployment_id)
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_get_deployment_info_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_get_deployment_info_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        For deployment '{deployment_id}', list a few of the feature names the model expects
        as inputs when scoring.
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
        For deployment '{deployment_id}', produce about two sample rows that match the shape
        needed for scoring, then describe the columns in those rows.
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
        For deployment '{deployment_id}', check whether this inline CSV is acceptable for scoring.
        Use this exact CSV text (verbatim, including header and newlines, no extra spaces):
        {csv}
        Summarize validation status and any warnings.
        """
        ],
    )
    async def test_validate_prediction_data_success(
        self,
        llm_client: Any,
        expectations_for_validate_prediction_data_success: ETETestExpectations,
        deployment_id: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(deployment_id=deployment_id, csv=INLINE_CSV_DATASET)
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
