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

import json
from typing import Any

from pydantic import BaseModel

from .clients.base import LLMResponse


class ToolCallTestExpectations(BaseModel):
    """Class to store tool call information."""

    name: str
    parameters: dict[str, Any]
    result: str | dict[str, Any]


class ETETestExpectations(BaseModel):
    """Class to store test expectations for ETE tests."""

    potential_no_tool_calls: bool = False
    tool_calls_expected: list[ToolCallTestExpectations]
    llm_response_content_contains_expectations: list[str]


SHOULD_NOT_BE_EMPTY = "SHOULD_NOT_BE_EMPTY"


def _extract_structured_content(tool_result: str) -> Any:
    r"""
    Extract and parse structured content from tool result string.

    Tool results are formatted as:
    "Content: {content}\nStructured content: {structured_content}"

    Structured content can be:
    1. A JSON object with a "result" key: {"result": "..."} or {"result": "{...}"}
    2. A direct JSON object: {"key": "value", ...}
    3. Empty or missing

    Args:
        tool_result: The tool result string

    Returns
    -------
        Parsed structured content, or None if not available
    """
    # Early returns for invalid inputs
    if not tool_result or "Structured content: " not in tool_result:
        return None

    structured_part = tool_result.split("Structured content: ", 1)[1].strip()
    # Parse JSON, return None on failure or empty structured_part
    if not structured_part:
        return None
    try:
        structured_data = json.loads(structured_part)
    except json.JSONDecodeError:
        return None

    # If structured data has a "result" key, extract and parse that
    if isinstance(structured_data, dict) and "result" in structured_data:
        result_value = structured_data["result"]
        # If result is a JSON string (starts with { or [), try to parse it
        if isinstance(result_value, str) and result_value.strip().startswith(("{", "[")):
            try:
                parsed_result = json.loads(result_value)
            except json.JSONDecodeError:
                parsed_result = result_value  # Return string as-is if parsing fails
            return parsed_result
        return result_value  # Return result value directly

    # If it's a direct JSON object (not wrapped in {"result": ...}), return it as-is
    return structured_data


def _check_dict_has_keys(
    expected: dict[str, Any],
    actual: dict[str, Any] | list[dict[str, Any]],
    path: str = "",
) -> bool:
    """
    Recursively check if all keys in expected dict exist in actual dict or in each item of
    actual list.
    Returns True if all expected keys exist, False otherwise.
    """
    # If actual is a list, check each item against the expected structure
    if isinstance(actual, list):
        if not actual:  # Empty list
            return False
        # Check first item against expected structure
        return _check_dict_has_keys(expected, actual[0], path)

    # Regular dict check
    for key, value in expected.items():
        current_path = f"{path}.{key}" if path else key
        if key not in actual:
            return False
        if isinstance(value, dict):
            if not isinstance(actual[key], dict):
                return False
            if not _check_dict_has_keys(value, actual[key], current_path):
                return False
    return True


class ToolBaseE2E:
    """Base class for end-to-end tests."""

    async def _run_test_with_expectations(
        self,
        prompt: str,
        test_expectations: ETETestExpectations,
        openai_llm_client: Any,
        mcp_session: Any,
        test_name: str,
    ) -> None:
        """
        Run a test with given expectations and validate the results.

        Args:
            prompt: The prompt to send to the LLM
            test_expectations: ETETestExpectations object containing test expectations with keys:
                - tool_calls_expected: List of expected tool calls with their parameters and results
                - llm_response_content_contains_expectations: Expected content in the LLM response
            openai_llm_client: The OpenAI LLM client
            mcp_session: The test session
            test_name: The name of the test (e.g. test_get_best_model_success)
        """
        # Get the test file name from the class name
        file_name = self.__class__.__name__.lower().replace("e2e", "").replace("test", "")
        output_file_name = f"{file_name}_{test_name}"

        # Act
        response: LLMResponse = await openai_llm_client.process_prompt_with_mcp_support(
            prompt, mcp_session, output_file_name
        )

        # sometimes llm are too smart and doesn't call tools especially for the case when file
        # doesn't exist
        if test_expectations.potential_no_tool_calls and len(response.tool_calls) == 0:
            pass
        else:
            # Verify LLM decided to use tools
            assert len(response.tool_calls) == len(test_expectations.tool_calls_expected), (
                "LLM should have decided to call tools"
            )

            for i, tool_call in enumerate(response.tool_calls):
                assert tool_call.tool_name == test_expectations.tool_calls_expected[i].name, (
                    f"Should have called {test_expectations.tool_calls_expected[i].name} tool, but "
                    f"got: {tool_call.tool_name}"
                )
                assert (
                    tool_call.parameters == test_expectations.tool_calls_expected[i].parameters
                ), (
                    f"Should have called {tool_call.tool_name} tool with the correct parameters, "
                    f"but got: {tool_call.parameters}"
                )
                if test_expectations.tool_calls_expected[i].result != SHOULD_NOT_BE_EMPTY:
                    expected_result = test_expectations.tool_calls_expected[i].result
                    if isinstance(expected_result, str):
                        assert expected_result in response.tool_results[i], (
                            f"Should have called {tool_call.tool_name} tool with the correct "
                            f"result, but got: {response.tool_results[i]}"
                        )
                    else:
                        actual_result = _extract_structured_content(response.tool_results[i])
                        if actual_result is None:
                            # Fallback: try to parse the entire tool result as JSON
                            try:
                                actual_result = json.loads(response.tool_results[i])
                            except json.JSONDecodeError:
                                # If that fails, try to extract content part
                                if "Content: " in response.tool_results[i]:
                                    content_part = response.tool_results[i].split("Content: ", 1)[1]
                                    if "\nStructured content: " in content_part:
                                        content_part = content_part.split(
                                            "\nStructured content: ", 1
                                        )[0]
                                    try:
                                        actual_result = json.loads(content_part.strip())
                                    except json.JSONDecodeError:
                                        raise AssertionError(
                                            f"Could not parse tool result for "
                                            f"{tool_call.tool_name}: {response.tool_results[i]}"
                                        )
                        assert _check_dict_has_keys(expected_result, actual_result), (
                            f"Should have called {tool_call.tool_name} tool with the correct "
                            f"result structure, but got: {response.tool_results[i]}"
                        )
                else:
                    assert len(response.tool_results[i]) > 0, (
                        f"Should have called {tool_call.tool_name} tool with non-empty result, but "
                        f"got: {response.tool_results[i]}"
                    )

        # Verify LLM provided comprehensive response
        assert len(response.content) > 100, "LLM should provide detailed response"
        assert any(
            expected_response.lower() in response.content
            for expected_response in test_expectations.llm_response_content_contains_expectations
        ), (
            f"Response should mention "
            f"{test_expectations.llm_response_content_contains_expectations}, "
            f"but got: {response.content}"
        )
