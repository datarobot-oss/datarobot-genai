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
import re
from typing import Any

from pydantic import BaseModel

from .clients.base import LLMResponse


class ToolCallTestExpectations(BaseModel):
    """Class to store tool call information."""

    name: str
    parameters: dict[str, Any]
    result: str | dict[str, Any]


class ETETestExpectations(BaseModel):
    """Class to store test expectations for ETE tests.

    By default ``allow_unexpected_tool_calls`` is True so models may call extra tools
    (e.g. list_projects after an error, get_dataset_details after resolving an id).
    Set it to False when a test must assert an exact tool-call count and order with
    no additional calls.
    """

    potential_no_tool_calls: bool = False
    allow_unexpected_tool_calls: bool = True
    tool_calls_expected: list[ToolCallTestExpectations]
    llm_response_content_contains_expectations: list[str]


SHOULD_NOT_BE_EMPTY = "SHOULD_NOT_BE_EMPTY"


class _AnyNonemptyStringSentinel:
    """Sentinel: expected param must be a non-blank string (e.g. job_id from a prior tool)."""


ANY_NONEMPTY_STRING = _AnyNonemptyStringSentinel()


def _truncate(text: str, max_len: int = 400) -> str:
    """Return a single-line truncated representation for failure diagnostics."""
    flattened = text.replace("\n", "\\n")
    if len(flattened) <= max_len:
        return flattened
    return f"{flattened[:max_len]}...<truncated>"


def _normalize_tool_name(tool_name: str, expected_tool_name: str | None = None) -> str:
    """Normalize namespaced MCP tool names to their base function name.

    Some environments expose tool names as `mcp_<server>_<tool_name>` while others expose
    plain `<tool_name>`. Acceptance checks should compare logical tool names, not namespace.

    When expected_tool_name is provided, match by suffix for namespaced MCP tools so server
    names that include underscores do not break normalization.
    """
    if expected_tool_name:
        if tool_name == expected_tool_name:
            return tool_name
        if tool_name.startswith("mcp_") and tool_name.endswith(f"_{expected_tool_name}"):
            return expected_tool_name

    # Some environments expose names as `<server>_mcp_<tool_name>`
    # (e.g. `global_mcp_upload_dataset_to_ai_catalog`). Strip that server prefix.
    if "_mcp_" in tool_name:
        _, suffix = tool_name.split("_mcp_", 1)
        if suffix:
            # If expected_tool_name is provided, validate the suffix matches
            if expected_tool_name is None or suffix == expected_tool_name:
                return suffix

    match = re.match(r"^mcp_[^_]+_(.+)$", tool_name)
    return match.group(1) if match else tool_name


def _extract_content_when_structured_empty(tool_result: str) -> dict[str, str] | None:
    r"""When Content is present but Structured content is empty, return {"error": content}."""
    if "Content: " not in tool_result:
        return None
    content_part = tool_result.split("Content: ", 1)[1]
    if "\nStructured content: " in content_part:
        content_part = content_part.split("\nStructured content: ", 1)[0]
    content_part = content_part.strip()
    return {"error": content_part} if content_part else None


def _extract_structured_content(tool_result: str) -> Any:
    r"""
    Extract and parse structured content from tool result string.

    Tool results are formatted as:
    "Content: {content}\nStructured content: {structured_content}"

    Structured content can be:
    1. A JSON object with a "result" key: {"result": "..."} or {"result": "{...}"}
    2. A direct JSON object: {"key": "value", ...}
    3. Empty or missing — when Content is present but Structured content is empty
       (e.g. tool errors), returns {"error": content} so dict expectations can validate.
    4. None if neither valid structured content nor Content is available

    Args:
        tool_result: The tool result string

    Returns
    -------
        Parsed structured content, or None if not available
    """
    result: Any = None
    if not tool_result:
        pass
    elif "Structured content: " in tool_result:
        structured_part = tool_result.split("Structured content: ", 1)[1].strip()
        if not structured_part:
            result = _extract_content_when_structured_empty(tool_result)
        else:
            try:
                structured_data = json.loads(structured_part)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(structured_data, dict) and "result" in structured_data:
                    result_value = structured_data["result"]
                    if isinstance(result_value, str) and result_value.strip().startswith(
                        ("{", "[")
                    ):
                        try:
                            result = json.loads(result_value)
                        except json.JSONDecodeError:
                            result = result_value
                    else:
                        result = result_value
                else:
                    result = structured_data
    else:
        result = _extract_content_when_structured_empty(tool_result)
    return result


def _param_leaf_matches(expected: Any, actual: Any) -> bool:
    """Return whether ``actual`` satisfies the expected leaf constraint."""
    if expected is ANY_NONEMPTY_STRING:
        return isinstance(actual, str) and bool(str(actual).strip())
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.strip() == expected.strip()
    return actual == expected


def _check_dict_params_match(
    expected: dict[str, Any],
    actual: dict[str, Any],
    path: str = "",
) -> bool:
    """
    Recursively check if all keys in expected exist in actual with matching values.
    Extra keys in actual are ignored (subset/partial matching).

    Use :data:`ANY_NONEMPTY_STRING` as an expected leaf when the value is any non-blank string
    (e.g. IDs from a prior tool). String literals compare with leading/trailing whitespace stripped.
    """
    for key, value in expected.items():
        current_path = f"{path}.{key}" if path else key
        if key not in actual:
            return False
        actual_v = actual[key]
        if isinstance(value, dict):
            if not isinstance(actual_v, dict) or not _check_dict_params_match(
                value, actual_v, current_path
            ):
                return False
        elif not _param_leaf_matches(value, actual_v):
            return False
    return True


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


def _build_failure_diagnostics(
    expected_calls: list[ToolCallTestExpectations],
    response: LLMResponse,
) -> str:
    """Build a compact diagnostics section for assertion errors."""
    expected_lines = [
        f"#{idx + 1} {call.name} params={call.parameters}"
        for idx, call in enumerate(expected_calls)
    ]
    actual_lines = []
    for idx, tool_call in enumerate(response.tool_calls):
        expected_name = expected_calls[idx].name if idx < len(expected_calls) else None
        normalized_name = _normalize_tool_name(tool_call.tool_name, expected_name)
        actual_lines.append(
            f"#{idx + 1} raw={tool_call.tool_name} "
            f"normalized={normalized_name} "
            f"params={tool_call.parameters}"
        )
    result_lines = [
        f"#{idx + 1} {_truncate(result)}" for idx, result in enumerate(response.tool_results)
    ]
    return (
        "Diagnostics:\n"
        f"Expected tool calls ({len(expected_calls)}):\n  "
        + "\n  ".join(expected_lines or ["<none>"])
        + f"\nActual tool calls ({len(response.tool_calls)}):\n  "
        + "\n  ".join(actual_lines or ["<none>"])
        + f"\nTool results ({len(response.tool_results)}):\n  "
        + "\n  ".join(result_lines or ["<none>"])
    )


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
                - allow_unexpected_tool_calls: Default True — expected calls must appear in order,
                  with optional extra calls allowed. Set False for strict exact count.
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
            diagnostics = _build_failure_diagnostics(
                test_expectations.tool_calls_expected,
                response,
            )

            expected_calls = test_expectations.tool_calls_expected
            expected_actual_indices: list[tuple[int, int]] = []

            # Verify LLM decided to use tools
            if test_expectations.allow_unexpected_tool_calls:
                assert len(response.tool_calls) >= len(expected_calls), (
                    f"LLM should have decided to call tools\n{diagnostics}"
                )
                search_start = 0
                for expected_idx, expected_call in enumerate(expected_calls):
                    matched_actual_idx = None
                    for actual_idx in range(search_start, len(response.tool_calls)):
                        actual_call = response.tool_calls[actual_idx]
                        actual_tool_name = _normalize_tool_name(
                            actual_call.tool_name, expected_call.name
                        )
                        if actual_tool_name != expected_call.name:
                            continue
                        if not _check_dict_params_match(
                            expected_call.parameters, actual_call.parameters
                        ):
                            continue
                        matched_actual_idx = actual_idx
                        break

                    assert matched_actual_idx is not None, (
                        f"Should have called {expected_call.name} tool with the correct "
                        f"parameters. Expected (subset): {expected_call.parameters}\n"
                        f"{diagnostics}"
                    )
                    expected_actual_indices.append((expected_idx, matched_actual_idx))
                    search_start = matched_actual_idx + 1
            else:
                assert len(response.tool_calls) == len(expected_calls), (
                    f"LLM should have decided to call tools\n{diagnostics}"
                )
                expected_actual_indices = [(i, i) for i in range(len(expected_calls))]

            for expected_idx, actual_idx in expected_actual_indices:
                tool_call = response.tool_calls[actual_idx]
                expected_tool_name = expected_calls[expected_idx].name
                actual_tool_name = _normalize_tool_name(tool_call.tool_name, expected_tool_name)

                assert actual_tool_name == expected_tool_name, (
                    f"Should have called {expected_tool_name} tool, but got: "
                    f"{tool_call.tool_name}\n"
                    f"{diagnostics}"
                )
                assert _check_dict_params_match(
                    expected_calls[expected_idx].parameters, tool_call.parameters
                ), (
                    f"Should have called {expected_tool_name} tool with the correct parameters. "
                    f"Expected (subset): {expected_calls[expected_idx].parameters}, "
                    f"but got: {tool_call.parameters}\n"
                    f"{diagnostics}"
                )
                if expected_calls[expected_idx].result != SHOULD_NOT_BE_EMPTY:
                    expected_result = expected_calls[expected_idx].result
                    if isinstance(expected_result, str):
                        assert expected_result in response.tool_results[actual_idx], (
                            f"Should have called {expected_tool_name} tool with the correct "
                            f"result, but got: {response.tool_results[actual_idx]}\n"
                            f"{diagnostics}"
                        )
                    else:
                        actual_result = _extract_structured_content(
                            response.tool_results[actual_idx]
                        )
                        if actual_result is None:
                            # Fallback: try to parse the entire tool result as JSON
                            try:
                                actual_result = json.loads(response.tool_results[actual_idx])
                            except json.JSONDecodeError:
                                # If that fails, try to extract content part
                                if "Content: " in response.tool_results[actual_idx]:
                                    content_part = response.tool_results[actual_idx].split(
                                        "Content: ", 1
                                    )[1]
                                    if "\nStructured content: " in content_part:
                                        content_part = content_part.split(
                                            "\nStructured content: ", 1
                                        )[0]
                                    try:
                                        actual_result = json.loads(content_part.strip())
                                    except json.JSONDecodeError:
                                        raise AssertionError(
                                            f"Could not parse tool result for "
                                            f"{expected_tool_name}: "
                                            f"{response.tool_results[actual_idx]}"
                                        )
                        assert _check_dict_has_keys(expected_result, actual_result), (
                            f"Should have called {expected_tool_name} tool with the correct "
                            f"result structure, but got: {response.tool_results[actual_idx]}\n"
                            f"{diagnostics}"
                        )
                else:
                    assert len(response.tool_results[actual_idx]) > 0, (
                        f"Should have called {expected_tool_name} tool with non-empty result, but "
                        f"got: {response.tool_results[actual_idx]}\n"
                        f"{diagnostics}"
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
