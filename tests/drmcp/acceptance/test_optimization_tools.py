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

"""Acceptance (E2E) tests for the cuopt_solve MCP tool.

These tests run against a live MCP server with an LLM and verify that the LLM
correctly invokes the cuopt_solve tool for optimization requests.

Requirements:
  - DR_MCP_SERVER_URL (or OPENAI_API_KEY + OPENAI_MODEL) configured
  - MCP server has ENABLE_OPTIMIZATION_TOOLS=true and CUOPT_DEPLOYMENT_ID set
"""

import inspect
from typing import Any

import pytest

from datarobot_genai.drmcp.test_utils.mcp_utils_ete import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY
from datarobot_genai.drmcp.test_utils.tool_base_ete import ETETestExpectations
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolBaseE2E
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolCallTestExpectations

_SIMPLE_LP_PROBLEM = {
    "type": "lp",
    "objective": {"minimize": ["x", "y"], "coefficients": [1.0, 2.0]},
    "constraints": [{"lhs": ["x", "y"], "coefficients": [1.0, 1.0], "rhs": 1.0, "sense": ">="}],
    "bounds": {"x": [0, None], "y": [0, None]},
}


@pytest.fixture(scope="session")
def expectations_for_cuopt_solve_success() -> ETETestExpectations:
    """Return expectations for a successful cuopt_solve LP solve call."""
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="cuopt_solve",
                parameters={"problem_definition": _SIMPLE_LP_PROBLEM},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "solution",
            "optimal",
            "objective",
            "result",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_cuopt_solve_preview_success() -> ETETestExpectations:
    """Return expectations for a cuopt_solve preview (validate) call."""
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="cuopt_solve",
                parameters={"problem_definition": _SIMPLE_LP_PROBLEM, "preview": True},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "valid",
            "preview",
            "problem",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_cuopt_solve_missing_deployment_error() -> ETETestExpectations:
    """Return expectations when CUOPT_DEPLOYMENT_ID is not configured."""
    return ETETestExpectations(
        potential_no_tool_calls=True,
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="cuopt_solve",
                parameters={"problem_definition": _SIMPLE_LP_PROBLEM},
                result="CUOPT_DEPLOYMENT_ID",
            ),
        ],
        llm_response_content_contains_expectations=[
            "deployment",
            "not configured",
            "environment",
        ],
    )


@pytest.mark.asyncio
class TestCuoptSolveE2E(ToolBaseE2E):
    """End-to-end tests for the cuopt_solve tool."""

    @pytest.mark.parametrize(
        "prompt_template",
        [
            "Please solve the following LP optimization problem using the cuopt_solve tool. "
            "The problem definition is: "
            "minimize x + 2y subject to x + y >= 1 with x, y >= 0. "
            "Pass the problem as a dictionary with type, objective, constraints, and bounds."
        ],
    )
    async def test_cuopt_solve_lp_success(
        self,
        llm_client: Any,
        expectations_for_cuopt_solve_success: ETETestExpectations,
        prompt_template: str,
    ) -> None:
        """Test that LLM correctly calls cuopt_solve with an LP problem definition."""
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_cuopt_solve_lp_success"
            await self._run_test_with_expectations(
                prompt_template,
                expectations_for_cuopt_solve_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            "Please validate (preview only, do not solve) the following LP optimization problem "
            "using the cuopt_solve tool with preview=True. "
            "The problem definition is: minimize x + 2y subject to x + y >= 1 with x, y >= 0."
        ],
    )
    async def test_cuopt_solve_preview_success(
        self,
        llm_client: Any,
        expectations_for_cuopt_solve_preview_success: ETETestExpectations,
        prompt_template: str,
    ) -> None:
        """Test that LLM correctly calls cuopt_solve with preview=True for validation."""
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_cuopt_solve_preview_success"
            await self._run_test_with_expectations(
                prompt_template,
                expectations_for_cuopt_solve_preview_success,
                llm_client,
                session,
                test_name,
            )
