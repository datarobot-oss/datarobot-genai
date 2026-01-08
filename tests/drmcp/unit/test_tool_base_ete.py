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

from datarobot_genai.drmcp.test_utils.tool_base_ete import ETETestExpectations
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolBaseE2E
from datarobot_genai.drmcp.test_utils.tool_base_ete import ToolCallTestExpectations
from datarobot_genai.drmcp.test_utils.tool_base_ete import _check_dict_has_keys


class TestToolCallTestExpectations:
    """Test cases for ToolCallTestExpectations class."""

    def test_tool_call_test_expectations_creation(self) -> None:
        """Test ToolCallTestExpectations creation."""
        expectations = ToolCallTestExpectations(
            name="test_tool", parameters={"param": "value"}, result="result"
        )

        assert expectations.name == "test_tool"
        assert expectations.parameters == {"param": "value"}
        assert expectations.result == "result"

    def test_tool_call_test_expectations_with_dict_result(self) -> None:
        """Test ToolCallTestExpectations with dict result."""
        expectations = ToolCallTestExpectations(
            name="test_tool", parameters={}, result={"status": "success"}
        )

        assert isinstance(expectations.result, dict)
        assert expectations.result["status"] == "success"


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

    def test_ete_test_expectations_with_potential_no_tool_calls(self) -> None:
        """Test ETETestExpectations with potential_no_tool_calls set."""
        expectations = ETETestExpectations(
            tool_calls_expected=[],
            llm_response_content_contains_expectations=[],
            potential_no_tool_calls=True,
        )

        assert expectations.potential_no_tool_calls is True


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
