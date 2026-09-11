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

"""Unit tests for tool_base_ete.py module."""

from datarobot_genai.drmcp.test_utils.clients.base import ToolCall
from datarobot_genai.drmcp.test_utils.tool_base_ete import ANY_NONEMPTY_STRING
from datarobot_genai.drmcp.test_utils.tool_base_ete import ETETestExpectations
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolBaseE2E
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolCallTestExpectations
from datarobot_genai.drmcp.test_utils.tool_base_ete import _canonical_tool_name_for_expectation
from datarobot_genai.drmcp.test_utils.tool_base_ete import _check_dict_has_keys
from datarobot_genai.drmcp.test_utils.tool_base_ete import _check_dict_params_match
from datarobot_genai.drmcp.test_utils.tool_base_ete import _forbidden_parameter_violations
from datarobot_genai.drmcp.test_utils.tool_base_ete import _forbidden_tool_call_violations


class TestToolCallTestExpectations:
    """Test cases for ToolCallTestExpectations class."""

    def test_tool_call_test_expectations_creation(self) -> None:
        """Test ToolCallTestExpectations creation."""
        expectations = ToolCallTestExpectations(
            name="test_tool", parameters={"param": "value"}, result="result"
        )

        assert expectations.name == "test_tool"
        assert expectations.acceptable_tool_names == []
        assert expectations.parameters == {"param": "value"}
        assert expectations.result == "result"

    def test_tool_call_test_expectations_allowed_tool_names(self) -> None:
        expectations = ToolCallTestExpectations(
            name="a",
            acceptable_tool_names=["b", "c"],
            parameters={},
            result="x",
        )
        assert expectations.allowed_tool_names() == {"a", "b", "c"}

    def test_tool_call_test_expectations_with_dict_result(self) -> None:
        """Test ToolCallTestExpectations with dict result."""
        expectations = ToolCallTestExpectations(
            name="test_tool", parameters={}, result={"status": "success"}
        )

        assert isinstance(expectations.result, dict)
        assert expectations.result["status"] == "success"

    def test_tool_call_test_expectations_forbidden_parameter_values_defaults_empty(self) -> None:
        """Test forbidden_parameter_values defaults to an empty dict."""
        expectations = ToolCallTestExpectations(name="test_tool", parameters={}, result="result")

        assert expectations.forbidden_parameter_values == {}

    def test_tool_call_test_expectations_forbidden_parameter_values_stores_value(self) -> None:
        """Test forbidden_parameter_values stores the value it is given."""
        expectations = ToolCallTestExpectations(
            name="otel_trace_get",
            parameters={"trace_id": "abc"},
            result="result",
            forbidden_parameter_values={"view": "payloads"},
        )

        assert expectations.forbidden_parameter_values == {"view": "payloads"}


class TestCanonicalToolNameForExpectation:
    def test_matches_primary_name(self) -> None:
        call = ToolCallTestExpectations(name="deployment_get_info", parameters={}, result="")
        assert (
            _canonical_tool_name_for_expectation("deployment_get_info", call)
            == "deployment_get_info"
        )

    def test_matches_acceptable_alternative(self) -> None:
        call = ToolCallTestExpectations(
            name="deployment_get_info",
            acceptable_tool_names=["deployment_get_features"],
            parameters={},
            result="",
        )
        assert (
            _canonical_tool_name_for_expectation("deployment_get_features", call)
            == "deployment_get_features"
        )

    def test_no_match(self) -> None:
        call = ToolCallTestExpectations(name="deployment_get_info", parameters={}, result="")
        assert _canonical_tool_name_for_expectation("other_tool", call) is None


class TestETETestExpectations:
    """Test cases for ETETestExpectations class."""

    def test_ete_test_expectations_creation(self) -> None:
        """Test ETETestExpectations creation."""
        tool_call = ToolCallTestExpectations(name="tool1", parameters={}, result="result1")
        expectations = ETETestExpectations(
            tool_calls_expected=[tool_call],
            llm_response_content_contains_expectations=["expected text"],
        )

        assert len(expectations.tool_calls_expected) == 1
        assert expectations.tool_calls_expected[0].name == "tool1"
        assert expectations.potential_no_tool_calls is False
        assert expectations.allow_unexpected_tool_calls is True

    def test_ete_test_expectations_with_potential_no_tool_calls(self) -> None:
        """Test ETETestExpectations with potential_no_tool_calls set."""
        expectations = ETETestExpectations(
            tool_calls_expected=[],
            llm_response_content_contains_expectations=[],
            potential_no_tool_calls=True,
        )

        assert expectations.potential_no_tool_calls is True

    def test_ete_test_expectations_forbidden_tool_names_defaults_empty(self) -> None:
        """Test forbidden_tool_names defaults to an empty list."""
        expectations = ETETestExpectations(
            tool_calls_expected=[],
            llm_response_content_contains_expectations=[],
        )

        assert expectations.forbidden_tool_names == []

    def test_ete_test_expectations_forbidden_tool_names_stores_value(self) -> None:
        """Test forbidden_tool_names stores the value it is given."""
        expectations = ETETestExpectations(
            tool_calls_expected=[],
            llm_response_content_contains_expectations=[],
            forbidden_tool_names=["otel_trace_get"],
        )

        assert expectations.forbidden_tool_names == ["otel_trace_get"]


class TestCheckDictHasKeys:
    """Test cases for _check_dict_has_keys function."""

    def test_check_dict_has_keys_simple_match(self) -> None:
        """Test _check_dict_has_keys with simple matching keys."""
        expected = {"key1": "value1", "key2": "value2"}
        actual = {"key1": "value1", "key2": "value2", "key3": "extra"}

        assert _check_dict_has_keys(expected, actual) is True

    def test_check_dict_has_keys_missing_key(self) -> None:
        """Test _check_dict_has_keys with missing key."""
        expected = {"key1": "value1", "key2": "value2"}
        actual = {"key1": "value1"}

        assert _check_dict_has_keys(expected, actual) is False

    def test_check_dict_has_keys_nested_dict(self) -> None:
        """Test _check_dict_has_keys with nested dictionaries."""
        expected = {"outer": {"inner": "value"}}
        actual = {"outer": {"inner": "value", "extra": "data"}}

        assert _check_dict_has_keys(expected, actual) is True

    def test_check_dict_has_keys_nested_missing_key(self) -> None:
        """Test _check_dict_has_keys with missing nested key."""
        expected = {"outer": {"inner": "value"}}
        actual = {"outer": {"other": "value"}}

        assert _check_dict_has_keys(expected, actual) is False

    def test_check_dict_has_keys_with_list(self) -> None:
        """Test _check_dict_has_keys with list of dicts."""
        expected = {"key1": "value1"}
        actual = [{"key1": "value1", "key2": "value2"}, {"key1": "value1"}]

        assert _check_dict_has_keys(expected, actual) is True

    def test_check_dict_has_keys_with_empty_list(self) -> None:
        """Test _check_dict_has_keys with empty list."""
        expected = {"key1": "value1"}
        actual = []

        assert _check_dict_has_keys(expected, actual) is False

    def test_check_dict_has_keys_list_with_missing_key(self) -> None:
        """Test _check_dict_has_keys with list containing dict missing key."""
        expected = {"key1": "value1", "key2": "value2"}
        actual = [{"key1": "value1"}]

        assert _check_dict_has_keys(expected, actual) is False

    def test_check_dict_has_keys_nested_in_list(self) -> None:
        """Test _check_dict_has_keys with nested dict in list."""
        expected = {"outer": {"inner": "value"}}
        actual = [{"outer": {"inner": "value", "extra": "data"}}]

        assert _check_dict_has_keys(expected, actual) is True

    def test_check_dict_has_keys_list_with_wrong_type(self) -> None:
        """Test _check_dict_has_keys when list item is not a dict."""
        expected = {"key1": "value1"}
        actual = ["not a dict"]

        assert _check_dict_has_keys(expected, actual) is False

    def test_check_dict_has_keys_nested_wrong_type(self) -> None:
        """Test _check_dict_has_keys when nested value is not a dict."""
        expected = {"outer": {"inner": "value"}}
        actual = {"outer": "not a dict"}

        assert _check_dict_has_keys(expected, actual) is False


class TestCheckDictParamsMatch:
    """Test cases for _check_dict_params_match function."""

    def test_exact_match_passes(self) -> None:
        """Test that exact match passes."""
        expected = {"key1": "value1", "key2": "value2"}
        actual = {"key1": "value1", "key2": "value2"}

        assert _check_dict_params_match(expected, actual) is True

    def test_subset_match_extra_actual_keys_passes(self) -> None:
        """Test that extra keys in actual are ignored."""
        expected = {"key1": "value1"}
        actual = {"key1": "value1", "key2": "extra", "key3": "also_extra"}

        assert _check_dict_params_match(expected, actual) is True

    def test_missing_expected_key_fails(self) -> None:
        """Test that missing expected key fails."""
        expected = {"key1": "value1", "key2": "value2"}
        actual = {"key1": "value1"}

        assert _check_dict_params_match(expected, actual) is False

    def test_wrong_value_fails(self) -> None:
        """Test that wrong value for expected key fails."""
        expected = {"key1": "value1"}
        actual = {"key1": "wrong_value"}

        assert _check_dict_params_match(expected, actual) is False

    def test_nested_dict_subset_match_passes(self) -> None:
        """Test nested dict with extra keys in actual passes."""
        expected = {"outer": {"inner": "value"}}
        actual = {"outer": {"inner": "value", "extra": "data"}, "other": "stuff"}

        assert _check_dict_params_match(expected, actual) is True

    def test_nested_dict_wrong_value_fails(self) -> None:
        """Test nested dict with wrong value fails."""
        expected = {"outer": {"inner": "value"}}
        actual = {"outer": {"inner": "wrong_value"}}

        assert _check_dict_params_match(expected, actual) is False

    def test_empty_expected_passes(self) -> None:
        """Test that empty expected dict passes regardless of actual."""
        expected: dict = {}
        actual = {"key1": "value1", "key2": "value2"}

        assert _check_dict_params_match(expected, actual) is True

    def test_nested_wrong_type_fails(self) -> None:
        """Test that wrong type for nested dict fails."""
        expected = {"outer": {"inner": "value"}}
        actual = {"outer": "not_a_dict"}

        assert _check_dict_params_match(expected, actual) is False

    def test_any_nonempty_string_accepts_nonblank(self) -> None:
        expected = {"job_id": ANY_NONEMPTY_STRING}
        assert _check_dict_params_match(expected, {"job_id": "abc-123"}) is True

    def test_any_nonempty_string_rejects_blank(self) -> None:
        expected = {"job_id": ANY_NONEMPTY_STRING}
        assert _check_dict_params_match(expected, {"job_id": ""}) is False
        assert _check_dict_params_match(expected, {"job_id": "   "}) is False


class TestForbiddenParameterViolations:
    """Test cases for _forbidden_parameter_violations function."""

    def test_no_violation_when_key_absent(self) -> None:
        """An omitted optional arg is not a violation, even if it defaults to that value."""
        forbidden = {"view": "payloads"}
        actual: dict = {"trace_id": "abc"}

        assert _forbidden_parameter_violations(forbidden, actual) == {}

    def test_no_violation_when_value_differs(self) -> None:
        """The call explicitly chose a different, acceptable value."""
        forbidden = {"view": "payloads"}
        actual = {"view": "summary"}

        assert _forbidden_parameter_violations(forbidden, actual) == {}

    def test_violation_when_value_matches(self) -> None:
        """The call explicitly chose the forbidden value."""
        forbidden = {"view": "payloads"}
        actual = {"view": "payloads", "trace_id": "abc"}

        assert _forbidden_parameter_violations(forbidden, actual) == {"view": "payloads"}

    def test_violation_string_comparison_strips_whitespace(self) -> None:
        """String comparison strips whitespace, matching _param_leaf_matches elsewhere."""
        forbidden = {"view": "payloads"}
        actual = {"view": "  payloads  "}

        assert _forbidden_parameter_violations(forbidden, actual) == {"view": "  payloads  "}

    def test_empty_forbidden_yields_no_violations(self) -> None:
        """An empty forbidden dict never reports a violation."""
        assert _forbidden_parameter_violations({}, {"view": "payloads"}) == {}

    def test_nested_dict_violation_recurses_with_subset_semantics(self) -> None:
        """A forbidden nested shape is flagged even when actual carries extra sibling fields.

        This mirrors ``parameters``' own subset matching (via ``_check_dict_params_match``) --
        the docstring's claim that nested values are "checked the same way `parameters` is
        checked" must actually hold, not just for the top level.
        """
        forbidden = {"filters": {"status": "error"}}
        actual = {"filters": {"status": "error", "extra": "x"}}

        assert _forbidden_parameter_violations(forbidden, actual) == {
            "filters": {"status": "error", "extra": "x"}
        }

    def test_nested_dict_no_violation_when_inner_value_differs(self) -> None:
        """A nested dict that does not match the forbidden subset is not a violation."""
        forbidden = {"filters": {"status": "error"}}
        actual = {"filters": {"status": "ok"}}

        assert _forbidden_parameter_violations(forbidden, actual) == {}

    def test_nested_dict_no_violation_when_actual_value_is_not_a_dict(self) -> None:
        """A forbidden nested dict never matches a scalar actual value at that key."""
        forbidden = {"filters": {"status": "error"}}
        actual = {"filters": "error"}

        assert _forbidden_parameter_violations(forbidden, actual) == {}


class TestForbiddenToolCallViolations:
    """Test cases for _forbidden_tool_call_violations function."""

    def test_empty_forbidden_names_yields_no_violations(self) -> None:
        """No forbidden names means nothing can violate, regardless of what was called."""
        calls = [ToolCall(tool_name="otel_trace_get", parameters={}, reasoning="")]

        assert _forbidden_tool_call_violations([], calls) == []

    def test_no_violation_when_forbidden_tool_never_called(self) -> None:
        """A call to an unrelated tool is not a violation."""
        calls = [ToolCall(tool_name="otel_span_payload_get", parameters={}, reasoning="")]

        assert _forbidden_tool_call_violations(["otel_trace_get"], calls) == []

    def test_violation_when_forbidden_tool_is_called(self) -> None:
        """A call to the forbidden tool is reported by its raw name."""
        calls = [ToolCall(tool_name="otel_trace_get", parameters={}, reasoning="")]

        assert _forbidden_tool_call_violations(["otel_trace_get"], calls) == ["otel_trace_get"]

    def test_violation_regardless_of_position_in_the_call_sequence(self) -> None:
        """The forbidden tool is caught even when it is not the first or only call.

        This is what lets a legitimate extra call to the *expected* tool (e.g. a paginated
        otel_span_payload_get continuation) coexist with a strict "never call X" assertion --
        the check scans every call rather than assuming a fixed position or count.
        """
        calls = [
            ToolCall(tool_name="otel_span_payload_get", parameters={}, reasoning=""),
            ToolCall(tool_name="otel_trace_get", parameters={}, reasoning=""),
            ToolCall(tool_name="otel_span_payload_get", parameters={}, reasoning=""),
        ]

        assert _forbidden_tool_call_violations(["otel_trace_get"], calls) == ["otel_trace_get"]

    def test_normalizes_namespaced_tool_names_before_matching(self) -> None:
        """A namespaced MCP tool name still matches its bare logical forbidden name."""
        calls = [ToolCall(tool_name="mcp_someserver_otel_trace_get", parameters={}, reasoning="")]

        assert _forbidden_tool_call_violations(["otel_trace_get"], calls) == [
            "mcp_someserver_otel_trace_get"
        ]


class TestToolBaseE2E:
    """Test cases for ToolBaseE2E class."""

    def test_tool_base_e2e_class_name_parsing(self) -> None:
        """Test that class name parsing works correctly."""

        class TestE2E(ToolBaseE2E):
            pass

        instance = TestE2E()
        file_name = instance.__class__.__name__.lower().replace("e2e", "").replace("test", "")

        assert file_name == ""

    def test_tool_base_e2e_class_name_with_test(self) -> None:
        """Test class name parsing with 'test' in name."""

        class TestDeploymentE2E(ToolBaseE2E):
            pass

        instance = TestDeploymentE2E()
        file_name = instance.__class__.__name__.lower().replace("e2e", "").replace("test", "")

        assert "deployment" in file_name

    def test_check_dict_has_keys_with_complex_nested_structure(self) -> None:
        """Test _check_dict_has_keys with complex nested structure."""
        expected = {
            "level1": {
                "level2": {
                    "level3": "value",
                },
            },
        }
        actual = {
            "level1": {
                "level2": {
                    "level3": "value",
                    "extra": "data",
                },
                "extra2": "data",
            },
            "extra3": "data",
        }

        assert _check_dict_has_keys(expected, actual) is True

    def test_check_dict_has_keys_with_missing_nested_key(self) -> None:
        """Test _check_dict_has_keys with missing nested key."""
        expected = {
            "level1": {
                "level2": {
                    "level3": "value",
                },
            },
        }
        actual = {
            "level1": {
                "level2": {
                    "other": "value",
                },
            },
        }

        assert _check_dict_has_keys(expected, actual) is False
