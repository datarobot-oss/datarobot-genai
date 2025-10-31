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
def expectations_for_list_projects_success(
    classification_project_id: str,
    classification_project_name: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="list_projects",
                parameters={},
                result=f"{classification_project_id}: ",
            ),
        ],
        llm_response_content_contains_expectations=[
            classification_project_id,
            classification_project_name,
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_get_project_dataset_by_name_success(
    classification_project_id: str,
    classification_dataset_name: str,
    classification_dataset_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="get_project_dataset_by_name",
                parameters={
                    "project_id": classification_project_id,
                    "dataset_name": classification_dataset_name,
                },
                result={
                    "dataset_id": classification_dataset_id,
                    "dataset_type": "source",
                    "ui_panel": ["dataset"],
                },
            ),
        ],
        llm_response_content_contains_expectations=[
            classification_dataset_name,
            "source",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_get_project_dataset_by_name_failure(
    classification_project_id: str, nonexistent_dataset_name: str
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="get_project_dataset_by_name",
                parameters={
                    "project_id": classification_project_id,
                    "dataset_name": nonexistent_dataset_name,
                },
                result=(
                    f"Dataset with name containing '{nonexistent_dataset_name}' not found in "
                    f"project {classification_project_id}."
                ),
            ),
        ],
        llm_response_content_contains_expectations=[
            "dataset exists in the project",
            "not found",
            nonexistent_dataset_name,
            classification_project_id,
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_get_project_dataset_by_name_success_with_multiple_calls(
    classification_project_name: str,
    classification_dataset_name: str,
    classification_project_id: str,
    classification_dataset_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="list_projects",
                parameters={},
                result=f"{classification_project_id}: ",
            ),
            ToolCallTestExpectations(
                name="get_project_dataset_by_name",
                parameters={
                    "project_id": classification_project_id,
                    "dataset_name": classification_dataset_name,
                },
                result={
                    "dataset_id": classification_dataset_id,
                    "dataset_type": "source",
                    "ui_panel": ["dataset"],
                },
            ),
        ],
        llm_response_content_contains_expectations=[
            classification_project_name,
            classification_dataset_name,
            classification_dataset_id,
            "Source",
        ],
    )


@pytest.mark.asyncio
class TestProjectsE2E(ToolBaseE2E):
    """End-to-end tests for project-related functionality."""

    @pytest.mark.parametrize(
        "prompt",
        [
            """
        I'm working on a machine learning project and I need to list all the projects I
        have access to. Can you help me list all the projects?
        """
        ],
    )
    async def test_list_projects_success(
        self,
        openai_llm_client: Any,
        expectations_for_list_projects_success: ETETestExpectations,
        prompt: str,
    ) -> None:
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_list_projects_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_list_projects_success,
                openai_llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I'm working on a machine learning project ID {project_id} and I need to get the
        dataset by name '{dataset_name}'. Can you help me get the dataset by name?
        """
        ],
    )
    async def test_get_project_dataset_by_name_success(
        self,
        openai_llm_client: Any,
        expectations_for_get_project_dataset_by_name_success: ETETestExpectations,
        classification_project_id: str,
        classification_dataset_name: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            project_id=classification_project_id,
            dataset_name=classification_dataset_name,
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = (
                frame.f_code.co_name if frame else "test_get_project_dataset_by_name_success"
            )
            await self._run_test_with_expectations(
                prompt,
                expectations_for_get_project_dataset_by_name_success,
                openai_llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I'm working on a machine learning project ID {project_id} and I need to get the
        dataset by name '{dataset_name}'. Can you help me get the dataset by name?
        """
        ],
    )
    async def test_get_project_dataset_by_name_failure(
        self,
        openai_llm_client: Any,
        expectations_for_get_project_dataset_by_name_failure: ETETestExpectations,
        classification_project_id: str,
        nonexistent_dataset_name: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            project_id=classification_project_id, dataset_name=nonexistent_dataset_name
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = (
                frame.f_code.co_name if frame else "test_get_project_dataset_by_name_failure"
            )
            await self._run_test_with_expectations(
                prompt,
                expectations_for_get_project_dataset_by_name_failure,
                openai_llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            """
        I'm working on a machine learning project {project_name} and I need to get the
        dataset by name '{dataset_name}'. Can you help me get the dataset by name?
        """
        ],
    )
    async def test_get_project_dataset_by_name_success_with_multiple_calls(
        self,
        openai_llm_client: Any,
        expectations_for_get_project_dataset_by_name_success_with_multiple_calls: (
            ETETestExpectations
        ),
        classification_project_name: str,
        classification_dataset_name: str,
        prompt_template: str,
    ) -> None:
        prompt = prompt_template.format(
            project_name=classification_project_name,
            dataset_name=classification_dataset_name,
        )

        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = (
                frame.f_code.co_name
                if frame
                else "test_get_project_dataset_by_name_success_with_multiple_calls"
            )
            await self._run_test_with_expectations(
                prompt,
                expectations_for_get_project_dataset_by_name_success_with_multiple_calls,
                openai_llm_client,
                session,
                test_name,
            )
