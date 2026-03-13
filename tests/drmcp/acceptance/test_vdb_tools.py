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

"""Acceptance tests for VDB MCP tools (list_vector_databases, query_vector_database)."""

import inspect
from typing import Any

import pytest

from datarobot_genai.drmcp import ETETestExpectations
from datarobot_genai.drmcp import ToolBaseE2E
from datarobot_genai.drmcp import ToolCallTestExpectations
from datarobot_genai.drmcp import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY


@pytest.fixture(scope="session")
def vdb_deployment_id() -> str:
    """Return a VDB deployment ID for acceptance tests."""
    return "vdb_deployment_id_ete"


@pytest.mark.asyncio
class TestVDBToolsE2E(ToolBaseE2E):
    """End-to-end acceptance tests for VDB MCP tools."""

    async def test_list_vector_databases_callable(self, llm_client: Any) -> None:
        """Smoke test: LLM is prompted to list VDBs; expect tool use and valid response."""
        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="list_vector_databases",
                    parameters={},
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["vector database", "VDB", "deployment"],
        )
        prompt = (
            "Please list all Vector Databases (VDBs) available in DataRobot. "
            "Use the list_vector_databases tool."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_list_vector_databases_callable"
            await self._run_test_with_expectations(
                prompt,
                expectations,
                llm_client,
                session,
                test_name,
            )

    async def test_query_vector_database_callable(
        self,
        llm_client: Any,
        vdb_deployment_id: str,
    ) -> None:
        """Smoke test: LLM is prompted to query a VDB; expect tool use and valid response."""
        expectations = ETETestExpectations(
            tool_calls_expected=[
                ToolCallTestExpectations(
                    name="query_vector_database",
                    parameters={
                        "deployment_id": vdb_deployment_id,
                        "query": "What is DataRobot?",
                    },
                    result=SHOULD_NOT_BE_EMPTY,
                ),
            ],
            llm_response_content_contains_expectations=["document", "result", "search"],
        )
        prompt = (
            f"Please query the Vector Database deployment with ID '{vdb_deployment_id}' "
            "for 'What is DataRobot?' using the query_vector_database tool."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_query_vector_database_callable"
            await self._run_test_with_expectations(
                prompt,
                expectations,
                llm_client,
                session,
                test_name,
            )
