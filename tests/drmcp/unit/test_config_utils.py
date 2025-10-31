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

import pytest
from pydantic_core import PydanticUseDefault

from datarobot_genai.drmcp.core.config_utils import (
    extract_datarobot_credential_runtime_param_payload,
)
from datarobot_genai.drmcp.core.config_utils import extract_datarobot_dict_runtime_param_payload
from datarobot_genai.drmcp.core.config_utils import extract_datarobot_runtime_param_payload


class TestExtractDataRobotRuntimeParamPayload:
    """Test cases for extract_datarobot_runtime_param_payload function."""

    def test_string_boolean_true(self):
        """Test parsing string boolean 'true'."""
        result = extract_datarobot_runtime_param_payload("true")
        assert result is True

    def test_string_boolean_false(self):
        """Test parsing string boolean 'false'."""
        result = extract_datarobot_runtime_param_payload("false")
        assert result is False

    def test_string_boolean_case_insensitive(self):
        """Test parsing string boolean with different cases."""
        assert extract_datarobot_runtime_param_payload("TRUE") is True
        assert extract_datarobot_runtime_param_payload("False") is False

    def test_string_number(self):
        """Test parsing string number."""
        result = extract_datarobot_runtime_param_payload("123")
        assert result == 123

    def test_string_float(self):
        """Test parsing string float."""
        pi_value = 3.14
        result = extract_datarobot_runtime_param_payload("3.14")
        assert result == pi_value

    def test_string_json_with_payload(self):
        """Test parsing JSON string with payload."""
        json_str = '{"type":"string","payload":"test_value"}'
        result = extract_datarobot_runtime_param_payload(json_str)
        assert result == "test_value"

    def test_string_json_with_payload_none(self):
        """Test parsing JSON string with null payload raises PydanticUseDefault."""
        json_str = '{"type":"string","payload":null}'
        with pytest.raises(PydanticUseDefault):
            extract_datarobot_runtime_param_payload(json_str)

    def test_string_json_plain_value(self):
        """Test parsing plain JSON value."""
        result = extract_datarobot_runtime_param_payload('"plain_string"')
        assert result == "plain_string"

    def test_string_invalid_json(self):
        """Test handling invalid JSON string."""
        result = extract_datarobot_runtime_param_payload("invalid json")
        assert result == "invalid json"

    def test_non_string_input(self):
        """Test handling non-string input."""
        result = extract_datarobot_runtime_param_payload(123)
        assert result == 123

    def test_dict_input(self):
        """Test handling dict input."""
        result = extract_datarobot_runtime_param_payload({"key": "value"})
        assert result == {"key": "value"}

    def test_list_input(self):
        """Test handling list input."""
        result = extract_datarobot_runtime_param_payload([1, 2, 3])
        assert result == [1, 2, 3]


class TestExtractDataRobotDictRuntimeParamPayload:
    """Test cases for extract_datarobot_dict_runtime_param_payload function."""

    def test_dict_with_payload_string(self):
        """Test dict with payload as JSON string."""
        input_dict = {"type": "string", "payload": '{"key": "value"}'}
        result = extract_datarobot_dict_runtime_param_payload(input_dict)
        assert result == {"key": "value"}

    def test_dict_with_payload_dict(self):
        """Test dict with payload as dict."""
        input_dict = {"type": "string", "payload": {"key": "value"}}
        result = extract_datarobot_dict_runtime_param_payload(input_dict)
        assert result == {"key": "value"}

    def test_dict_with_payload_none(self):
        """Test dict with null payload raises PydanticUseDefault."""
        input_dict = {"type": "string", "payload": None}
        with pytest.raises(PydanticUseDefault):
            extract_datarobot_dict_runtime_param_payload(input_dict)

    def test_dict_with_invalid_json_payload(self):
        """Test dict with invalid JSON payload returns empty dict."""
        input_dict = {"type": "string", "payload": "invalid json"}
        result = extract_datarobot_dict_runtime_param_payload(input_dict)
        assert result == {}

    def test_dict_without_payload(self):
        """Test dict without payload key returns as-is."""
        input_dict = {"key": "value"}
        result = extract_datarobot_dict_runtime_param_payload(input_dict)
        assert result == {"key": "value"}

    def test_string_json_with_payload_string(self):
        """Test JSON string with payload as JSON string."""
        json_str = '{"type":"string","payload":"{\\"key\\":\\"value\\"}"}'
        result = extract_datarobot_dict_runtime_param_payload(json_str)
        assert result == {"key": "value"}

    def test_string_json_with_payload_dict(self):
        """Test JSON string with payload as dict."""
        json_str = '{"type":"string","payload":{"key":"value"}}'
        result = extract_datarobot_dict_runtime_param_payload(json_str)
        assert result == {"key": "value"}

    def test_string_json_with_payload_none(self):
        """Test JSON string with null payload raises PydanticUseDefault."""
        json_str = '{"type":"string","payload":null}'
        with pytest.raises(PydanticUseDefault):
            extract_datarobot_dict_runtime_param_payload(json_str)

    def test_string_json_with_invalid_payload_json(self):
        """Test JSON string with invalid payload JSON returns empty dict."""
        json_str = '{"type":"string","payload":"invalid json"}'
        result = extract_datarobot_dict_runtime_param_payload(json_str)
        assert result == {}

    def test_string_json_plain_dict(self):
        """Test JSON string that's already a dict."""
        json_str = '{"key":"value"}'
        result = extract_datarobot_dict_runtime_param_payload(json_str)
        assert result == {"key": "value"}

    def test_string_invalid_json(self):
        """Test invalid JSON string returns empty dict."""
        result = extract_datarobot_dict_runtime_param_payload("invalid json")
        assert result == {}

    def test_non_string_non_dict_input(self):
        """Test non-string, non-dict input returns empty dict."""
        result = extract_datarobot_dict_runtime_param_payload(123)
        assert result == {}


class TestExtractDataRobotCredentialRuntimeParamPayload:
    """Test cases for extract_datarobot_credential_runtime_param_payload function."""

    def test_dict_with_payload(self):
        """Test dict with payload."""
        input_dict = {"type": "credential", "payload": {"username": "user", "password": "pass"}}
        result = extract_datarobot_credential_runtime_param_payload(input_dict)
        assert result == {"username": "user", "password": "pass"}

    def test_dict_with_payload_none(self):
        """Test dict with null payload raises PydanticUseDefault."""
        input_dict = {"type": "credential", "payload": None}
        with pytest.raises(PydanticUseDefault):
            extract_datarobot_credential_runtime_param_payload(input_dict)

    def test_dict_without_payload(self):
        """Test dict without payload key returns as-is."""
        input_dict = {"username": "user", "password": "pass"}
        result = extract_datarobot_credential_runtime_param_payload(input_dict)
        assert result == {"username": "user", "password": "pass"}

    def test_string_json_with_payload(self):
        """Test JSON string with payload."""
        json_str = '{"type":"credential","payload":{"username":"user","password":"pass"}}'
        result = extract_datarobot_credential_runtime_param_payload(json_str)
        assert result == {"username": "user", "password": "pass"}

    def test_string_json_with_payload_none(self):
        """Test JSON string with null payload raises PydanticUseDefault."""
        json_str = '{"type":"credential","payload":null}'
        with pytest.raises(PydanticUseDefault):
            extract_datarobot_credential_runtime_param_payload(json_str)

    def test_string_json_plain_dict(self):
        """Test JSON string that's already a dict."""
        json_str = '{"username":"user","password":"pass"}'
        result = extract_datarobot_credential_runtime_param_payload(json_str)
        assert result == {"username": "user", "password": "pass"}

    def test_string_invalid_json(self):
        """Test invalid JSON string returns as-is."""
        result = extract_datarobot_credential_runtime_param_payload("invalid json")
        assert result == "invalid json"

    def test_non_string_non_dict_input(self):
        """Test non-string, non-dict input returns as-is."""
        result = extract_datarobot_credential_runtime_param_payload(123)
        assert result == 123
