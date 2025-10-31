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
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY
from datarobot_genai.drmcp.test_utils.tool_base_ete import ETETestExpectations
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolBaseE2E
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolCallTestExpectations


@pytest.fixture(scope="session")
def expectations_for_predict_realtime_file_success(
    deployment_id: str, classification_predict_file_path: Path
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="predict_realtime",
                parameters={
                    "deployment_id": deployment_id,
                    "file_path": str(classification_predict_file_path),
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "prediction",
        ],
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


@pytest.mark.asyncio
class TestPredictRealtimeE2E(ToolBaseE2E):
    """End-to-end tests for realtime prediction functionality."""

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I have a DataRobot deployment with ID '{deployment_id}'.
        Please run realtime predictions using the local CSV file at '{file_path}'.
        """
        ],
    )
    async def test_predict_realtime_file_success(
        self,
        openai_llm_client: Any,
        expectations_for_predict_realtime_file_success: ETETestExpectations,
        deployment_id: str,
        classification_predict_file_path: Path,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            deployment_id=deployment_id,
            file_path=str(classification_predict_file_path),
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_predict_realtime_file_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_predict_realtime_file_success,
                openai_llm_client,
                session,
                test_name,
            )

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
        openai_llm_client: Any,
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
                openai_llm_client,
                session,
                test_name,
            )
