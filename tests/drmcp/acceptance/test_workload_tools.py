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

"""Acceptance tests for DataRobot Workload API MCP tools."""

import inspect
import os
from typing import Any

import pytest

from datarobot_genai.drmcp import ETETestExpectations
from datarobot_genai.drmcp import ToolBaseE2E
from datarobot_genai.drmcp import ToolCallTestExpectations
from datarobot_genai.drmcp import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY


@pytest.fixture(scope="session")
def expectations_for_workload_list_success() -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="workload_list",
                parameters={},
                result={"count": 0, "offset": 0, "limit": 100},
            ),
        ],
        llm_response_content_contains_expectations=[
            "workload",
            "workloads",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_bundle_list_success() -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="bundle_list",
                parameters={},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "bundle",
            "cpu",
            "resource",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_artifact_get_list_success() -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="artifact_get",
                parameters={},
                result={"count": 0, "offset": 0, "limit": 100},
            ),
        ],
        llm_response_content_contains_expectations=[
            "artifact",
            "artifacts",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_workload_get_success(workload_id: str) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="workload_get",
                parameters={"workload_id": workload_id},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "workload",
            "status",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_workload_get_not_found(
    nonexistent_workload_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="workload_get",
                parameters={"workload_id": nonexistent_workload_id},
                result="[not_found] DataRobot API error (404):",
            ),
        ],
        llm_response_content_contains_expectations=[
            "not found",
            "does not exist",
            "unable",
            nonexistent_workload_id,
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_workload_stats_success(workload_id: str) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="workload_stats",
                parameters={"workload_id": workload_id},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "workload",
            "request",
            "stats",
        ],
    )


@pytest.mark.skipif(
    not os.getenv("ENABLE_WORKLOAD_TOOLS"),
    reason="Workload tools are not enabled on the MCP server",
)
@pytest.mark.asyncio
class TestWorkloadToolsE2E(ToolBaseE2E):
    """End-to-end acceptance tests for Workload API MCP tools."""

    async def test_workload_list_success(
        self,
        llm_client: Any,
        expectations_for_workload_list_success: ETETestExpectations,
    ) -> None:
        """LLM lists workloads via workload_list."""
        prompt = (
            "List all DataRobot workloads I have access to. "
            "Use the workload_list tool and summarize what you find."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_workload_list_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_workload_list_success,
                llm_client,
                session,
                test_name,
            )

    async def test_bundle_list_success(
        self,
        llm_client: Any,
        expectations_for_bundle_list_success: ETETestExpectations,
    ) -> None:
        """LLM lists compute bundles via bundle_list."""
        prompt = (
            "Show me the available compute resource bundles for DataRobot workloads. "
            "Use the bundle_list tool."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_bundle_list_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_bundle_list_success,
                llm_client,
                session,
                test_name,
            )

    async def test_artifact_get_list_success(
        self,
        llm_client: Any,
        expectations_for_artifact_get_list_success: ETETestExpectations,
    ) -> None:
        """LLM lists artifacts via artifact_get."""
        prompt = (
            "List all artifacts in DataRobot using the artifact_get tool "
            "(omit artifact_id to list). Summarize the results."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_artifact_get_list_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_artifact_get_list_success,
                llm_client,
                session,
                test_name,
            )

    async def test_workload_get_success(
        self,
        llm_client: Any,
        workload_id: str,
        expectations_for_workload_get_success: ETETestExpectations,
    ) -> None:
        """LLM fetches a single workload via workload_get."""
        prompt = (
            f"Get the details for DataRobot workload '{workload_id}' using workload_get. "
            "Tell me its name and status."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_workload_get_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_workload_get_success,
                llm_client,
                session,
                test_name,
            )

    async def test_workload_get_not_found(
        self,
        llm_client: Any,
        nonexistent_workload_id: str,
        expectations_for_workload_get_not_found: ETETestExpectations,
    ) -> None:
        """LLM handles a missing workload id from workload_get."""
        prompt = (
            f"Fetch workload '{nonexistent_workload_id}' with workload_get and explain "
            "what happened if it cannot be found."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_workload_get_not_found"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_workload_get_not_found,
                llm_client,
                session,
                test_name,
            )

    async def test_workload_stats_success(
        self,
        llm_client: Any,
        workload_id: str,
        expectations_for_workload_stats_success: ETETestExpectations,
    ) -> None:
        """LLM reads workload performance stats via workload_stats."""
        prompt = (
            f"Get performance statistics for workload '{workload_id}' using workload_stats. "
            "Summarize request volume or error rate if present."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_workload_stats_success"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_workload_stats_success,
                llm_client,
                session,
                test_name,
            )
