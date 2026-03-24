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

"""Acceptance (E2E) tests for the execute_code MCP tool.

These tests run against a live MCP server with an LLM and verify that the LLM
correctly invokes the execute_code tool for code execution requests.

Requirements:
  - DR_MCP_SERVER_URL (or OPENAI_API_KEY + OPENAI_MODEL) configured
  - MCP server has ENABLE_CODE_EXECUTION_TOOLS=true
"""

import inspect
from typing import Any

import pytest

from datarobot_genai.drmcp.test_utils.mcp_utils_ete import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY
from datarobot_genai.drmcp.test_utils.tool_base_ete import ETETestExpectations
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolBaseE2E
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolCallTestExpectations


@pytest.fixture(scope="session")
def expectations_for_execute_code_print_success() -> ETETestExpectations:
    """Return expectations for an execute_code print tool call."""
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="execute_code",
                parameters={"code": "print('Hello, World!')"},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "hello",
            "world",
            "not available",
            "code",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_execute_code_arithmetic_success() -> ETETestExpectations:
    """Return expectations for an execute_code arithmetic tool call."""
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="execute_code",
                parameters={"code": "result = 2 + 2"},
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "4",
            "not available",
            "code",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_execute_code_no_code_error() -> ETETestExpectations:
    """Return expectations when execute_code is invoked without code."""
    return ETETestExpectations(
        potential_no_tool_calls=True,
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="execute_code",
                parameters={},
                result="Code must be provided",
            ),
        ],
        llm_response_content_contains_expectations=[
            "code",
            "provide",
            "required",
        ],
    )


@pytest.mark.asyncio
class TestExecuteCodeE2E(ToolBaseE2E):
    """End-to-end tests for the execute_code tool."""

    @pytest.mark.parametrize(
        "prompt_template",
        ["Please run this Python code using the execute_code tool: print('Hello, World!')"],
    )
    async def test_execute_code_print_success(
        self,
        llm_client: Any,
        expectations_for_execute_code_print_success: ETETestExpectations,
        prompt_template: str,
    ) -> None:
        """Test that LLM correctly calls execute_code with a print statement."""
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_execute_code_print_success"
            await self._run_test_with_expectations(
                prompt_template,
                expectations_for_execute_code_print_success,
                llm_client,
                session,
                test_name,
            )

    @pytest.mark.parametrize(
        "prompt_template",
        [
            "Use the execute_code tool to compute 2 + 2 in Python and store the result in a "
            "variable named 'result'."
        ],
    )
    async def test_execute_code_arithmetic_success(
        self,
        llm_client: Any,
        expectations_for_execute_code_arithmetic_success: ETETestExpectations,
        prompt_template: str,
    ) -> None:
        """Test that LLM correctly calls execute_code with arithmetic code."""
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_execute_code_arithmetic_success"
            await self._run_test_with_expectations(
                prompt_template,
                expectations_for_execute_code_arithmetic_success,
                llm_client,
                session,
                test_name,
            )
