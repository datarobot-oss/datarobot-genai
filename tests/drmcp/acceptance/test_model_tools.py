# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Acceptance tests for get_model_details, is_eligible_for_timeseries_training tools."""

import inspect
from typing import Any

import pytest

from datarobot_genai.drmcp import ETETestExpectations
from datarobot_genai.drmcp import ToolBaseE2E
from datarobot_genai.drmcp import ToolCallTestExpectations
from datarobot_genai.drmcp import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY


@pytest.mark.asyncio
class TestGetModelDetailsE2E(ToolBaseE2E):
    """End-to-end acceptance tests for get_model_details tool."""

    async def test_get_model_details_callable(
        self,
        llm_client: Any,
        classification_project_id: str,
        model_id: str,
    ) -> None:
        """Smoke test: LLM calls get_model_details and returns model info."""
        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="get_model_details",
                    parameters={
                        "project_id": classification_project_id,
                        "model_id": model_id,
                    },
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["model", "feature"],
        )
        prompt = (
            f"I have a DataRobot project with ID '{classification_project_id}' and a model "
            f"with ID '{model_id}'. Please get the detailed information about this model, "
            "including feature impact. What features are most important?"
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_get_model_details_callable"
            await self._run_test_with_expectations(
                prompt,
                expectations,
                llm_client,
                session,
                test_name,
            )

    async def test_get_model_details_with_roc_curve(
        self,
        llm_client: Any,
        classification_project_id: str,
        model_id: str,
    ) -> None:
        """LLM calls get_model_details with ROC curve and returns curve data."""
        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="get_model_details",
                    parameters={
                        "project_id": classification_project_id,
                        "model_id": model_id,
                        "include_roc_curve": True,
                    },
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["model", "roc", "curve", "validation"],
        )
        prompt = (
            f"For the DataRobot model with ID '{model_id}' in project "
            f"'{classification_project_id}', get its details including the ROC curve data. "
            "Describe the model's performance."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_get_model_details_with_roc_curve"
            await self._run_test_with_expectations(
                prompt,
                expectations,
                llm_client,
                session,
                test_name,
            )


@pytest.mark.asyncio
class TestIsEligibleForTimeseriesTrainingE2E(ToolBaseE2E):
    """End-to-end acceptance tests for is_eligible_for_timeseries_training tool."""

    async def test_timeseries_eligibility_check(
        self,
        llm_client: Any,
        classification_dataset_id: str,
    ) -> None:
        """LLM calls is_eligible_for_timeseries_training and reports eligibility."""
        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="is_eligible_for_timeseries_training",
                    parameters={
                        "dataset_id": classification_dataset_id,
                        "datetime_column": "date",
                        "target_column": "sales",
                    },
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["eligible", "dataset", "time"],
        )
        prompt = (
            f"I want to check if dataset '{classification_dataset_id}' is eligible for "
            "time series training in DataRobot. The datetime column is 'date' and the "
            "target column is 'sales'. Please check eligibility and explain any issues."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_timeseries_eligibility_check"
            await self._run_test_with_expectations(
                prompt,
                expectations,
                llm_client,
                session,
                test_name,
            )
