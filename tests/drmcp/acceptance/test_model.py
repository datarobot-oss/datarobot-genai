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
def expectations_for_models_get_bestmodel_success(
    classification_project_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="models_get_bestmodel",
                parameters={"project_id": classification_project_id},
                result={
                    "project_id": "",
                    "best_model": {"id": "", "model_type": "", "metrics": {}},
                },
            ),
        ],
        llm_response_content_contains_expectations=[
            "Keras",
            "AUC",
            "Accuracy",
            "Balanced Accuracy",
            "FVE Multinomial",
            "LogLoss",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_models_get_bestmodel_failure(
    nonexistent_project_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="models_get_bestmodel",
                parameters={"project_id": nonexistent_project_id},
                result="[not_found] DataRobot API error (404):",
            ),
        ],
        llm_response_content_contains_expectations=[
            "Project not found",
            "not valid",
            "does not exist",
            "unable to",
            "not found",
            nonexistent_project_id,
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_modeling_score_dataset_success(
    classification_project_id: str,
    model_id: str,
    classification_predict_dataset: Any,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="modeling_score_dataset",
                parameters={
                    "project_id": classification_project_id,
                    "model_id": model_id,
                    "dataset_id": classification_predict_dataset["dataset_id"],
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "scoring",
            "job",
            "started",
            "dataset",
            "project",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_modeling_score_dataset_failure(
    classification_project_id: str,
    nonexistent_model_id: str,
    classification_predict_dataset: Any,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="modeling_score_dataset",
                parameters={
                    "project_id": classification_project_id,
                    "model_id": nonexistent_model_id,
                    "dataset_id": classification_predict_dataset["dataset_id"],
                },
                result="[not_found] DataRobot API error (404):",
            ),
        ],
        llm_response_content_contains_expectations=[
            "error",
            "does not exist",
            "not found",
            nonexistent_model_id,
            "issue",
            "invalid",
            "not valid",
            "provide a valid model ID",
        ],
    )


@pytest.mark.asyncio
class TestModelE2E(ToolBaseE2E):
    """End-to-end tests for model-related functionality."""

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I'm working on a machine learning project and I have a DataRobot project with ID
        '{project_id}'. I need to find out the best performing model in this project.
        Can you help me identify the top model and tell me about its performance metrics?
        """
        ],
    )
    async def test_models_get_bestmodel_success(
        self,
        llm_client: Any,
        expectations_for_models_get_bestmodel_success: ETETestExpectations,
        classification_project_id: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(project_id=classification_project_id)

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_models_get_bestmodel_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_models_get_bestmodel_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I'm working on a machine learning project and I have a DataRobot project with ID
        '{project_id}'. I need to find out which is the best performing model in this project.
        Can you help me identify the top model and tell me about its performance metrics?
        """
        ],
    )
    async def test_models_get_bestmodel_failure(
        self,
        llm_client: Any,
        expectations_for_models_get_bestmodel_failure: ETETestExpectations,
        nonexistent_project_id: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(project_id=nonexistent_project_id)

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_models_get_bestmodel_failure"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_models_get_bestmodel_failure,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I'm working on a machine learning project with ID '{project_id}' and I have a
        DataRobot model with ID '{model_id}'. Please start scoring the AI Catalog dataset
        '{dataset_id}' with that model (use the catalog dataset id, not a URL or local file).
        Can you help me score the dataset?
        """
        ],
    )
    async def test_modeling_score_dataset_success(
        self,
        llm_client: Any,
        expectations_for_modeling_score_dataset_success: ETETestExpectations,
        classification_project_id: str,
        model_id: str,
        classification_predict_dataset: Any,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            project_id=classification_project_id,
            model_id=model_id,
            dataset_id=classification_predict_dataset["dataset_id"],
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_modeling_score_dataset_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_modeling_score_dataset_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I'm working on a machine learning project with ID '{project_id}' but my model ID is
        wrong: '{model_id}'. Try to start scoring AI Catalog dataset '{dataset_id}' with that
        model id so we can see the error from the invalid model id.
        """
        ],
    )
    async def test_modeling_score_dataset_failure(
        self,
        llm_client: Any,
        expectations_for_modeling_score_dataset_failure: ETETestExpectations,
        classification_project_id: str,
        nonexistent_model_id: str,
        classification_predict_dataset: Any,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            project_id=classification_project_id,
            model_id=nonexistent_model_id,
            dataset_id=classification_predict_dataset["dataset_id"],
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_modeling_score_dataset_failure"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_modeling_score_dataset_failure,
                llm_client,
                session,
                test_name,
            )
