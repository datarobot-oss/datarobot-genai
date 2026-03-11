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

"""Acceptance tests for new data tools (get_dataset_details, list_datastores, browse_datastore, query_datastore)."""  # noqa: E501

import inspect
from typing import Any

import pytest

from datarobot_genai.drmcp import ETETestExpectations
from datarobot_genai.drmcp import ToolBaseE2E
from datarobot_genai.drmcp import ToolCallTestExpectations
from datarobot_genai.drmcp import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY


@pytest.fixture(scope="session")
def expectations_for_get_dataset_details(
    classification_dataset_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="get_dataset_details",
                parameters={"dataset_id": classification_dataset_id},
                result={"id": classification_dataset_id, "name": ""},
            ),
        ],
        llm_response_content_contains_expectations=[
            "dataset",
            "columns",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_list_datastores() -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="list_datastores",
                parameters={},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "datastore",
            "data",
        ],
    )


@pytest.mark.asyncio
class TestDataToolsE2E(ToolBaseE2E):
    """Acceptance tests for data tools."""

    async def test_get_dataset_details(
        self,
        llm_client: Any,
        expectations_for_get_dataset_details: ETETestExpectations,
        classification_dataset_id: str,
    ) -> None:
        """Test LLM can use get_dataset_details to retrieve dataset metadata."""
        prompt = (
            f"I have a dataset with ID '{classification_dataset_id}' in the DataRobot AI Catalog. "
            "Can you get me the details about this dataset including its columns and some sample rows?"
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_get_dataset_details"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_get_dataset_details,
                llm_client,
                session,
                test_name,
            )

    async def test_list_datastores(
        self,
        llm_client: Any,
        expectations_for_list_datastores: ETETestExpectations,
    ) -> None:
        """Test LLM can use list_datastores to show available data connections."""
        prompt = (
            "Can you list all available data connections (datastores) in my DataRobot environment? "
            "I need to see what databases are connected."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_list_datastores"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_list_datastores,
                llm_client,
                session,
                test_name,
            )
