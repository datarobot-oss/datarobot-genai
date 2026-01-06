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

"""Unit tests for test_utils/utils.py module."""

import json
from unittest.mock import mock_open
from unittest.mock import patch

from datarobot_genai.drmcp.test_utils.openai_llm_mcp_client import LLMResponse
from datarobot_genai.drmcp.test_utils.openai_llm_mcp_client import ToolCall
from datarobot_genai.drmcp.test_utils.utils import format_response
from datarobot_genai.drmcp.test_utils.utils import format_tool_call
from datarobot_genai.drmcp.test_utils.utils import load_env
from datarobot_genai.drmcp.test_utils.utils import save_response_to_file


class TestFormatToolCall:
    """Test cases for format_tool_call function."""

    def test_format_tool_call_basic(self) -> None:
        """Test format_tool_call with basic tool call."""
        tool_call = {
            "tool_name": "test_tool",
            "parameters": {"param1": "value1"},
            "reasoning": "test reasoning",
        }

        result = format_tool_call(tool_call)

        assert "test_tool" in result
        assert "param1" in result
        assert "value1" in result
        assert "test reasoning" in result

    def test_format_tool_call_with_complex_parameters(self) -> None:
        """Test format_tool_call with complex nested parameters."""
        tool_call = {
            "tool_name": "complex_tool",
            "parameters": {"nested": {"key": "value"}},
            "reasoning": "complex reasoning",
        }

        result = format_tool_call(tool_call)

        assert "complex_tool" in result
        assert "nested" in result
        assert "key" in result
        assert "value" in result


class TestFormatResponse:
    """Test cases for format_response function."""

    def test_format_response_without_tools(self) -> None:
        """Test format_response without tool calls."""
        response = LLMResponse(content="Simple response", tool_calls=[], tool_results=[])

        result = format_response(response)

        assert "Simple response" in result
        assert "=== LLM Response ===" in result
        assert "Tools Used" not in result

    def test_format_response_with_tools(self) -> None:
        """Test format_response with tool calls."""
        tool_calls = [
            ToolCall("tool1", {"param": "value"}, "reasoning1"),
            ToolCall("tool2", {}, "reasoning2"),
        ]
        tool_results = ["result1", "result2"]

        response = LLMResponse(
            content="Response with tools", tool_calls=tool_calls, tool_results=tool_results
        )

        result = format_response(response)

        assert "Response with tools" in result
        assert "Tools Used" in result
        assert "Tool Results" in result
        assert "tool1" in result
        assert "tool2" in result
        assert "result1" in result
        assert "result2" in result

    def test_format_response_with_tools_no_results(self) -> None:
        """Test format_response with tool calls but no results."""
        tool_calls = [ToolCall("tool1", {}, "reasoning")]
        response = LLMResponse(content="Response", tool_calls=tool_calls, tool_results=[])

        result = format_response(response)

        assert "tool1" in result
        # Tool Results section is only added when tool_results is truthy
        assert "Tool Results" not in result


class TestSaveResponseToFile:
    """Test cases for save_response_to_file function."""

    @patch("datarobot_genai.drmcp.test_utils.utils.os.makedirs")
    @patch("datarobot_genai.drmcp.test_utils.utils.open", new_callable=mock_open)
    @patch("datarobot_genai.drmcp.test_utils.utils.datetime")
    def test_save_response_to_file_with_name(self, mock_datetime, mock_file, mock_makedirs) -> None:
        """Test save_response_to_file with a name."""
        mock_datetime.datetime.now.return_value.strftime.return_value = "20250101"

        response = LLMResponse(content="Test response", tool_calls=[], tool_results=[])

        save_response_to_file(response, name="test_response")

        mock_makedirs.assert_called_once()
        mock_file.assert_called_once()
        # Verify file was written
        assert mock_file().write.called

    @patch("datarobot_genai.drmcp.test_utils.utils.os.makedirs")
    @patch("datarobot_genai.drmcp.test_utils.utils.open", new_callable=mock_open)
    @patch("datarobot_genai.drmcp.test_utils.utils.datetime")
    def test_save_response_to_file_without_name(
        self, mock_datetime, mock_file, mock_makedirs
    ) -> None:
        """Test save_response_to_file without a name."""
        mock_datetime.datetime.now.return_value.strftime.return_value = "20250101"

        response = LLMResponse(content="Test response", tool_calls=[], tool_results=[])

        save_response_to_file(response)

        mock_makedirs.assert_called_once()
        mock_file.assert_called_once()
        # Verify file path contains "response"
        call_args = mock_file.call_args[0][0]
        assert "response.txt" in call_args


class TestLoadEnv:
    """Test cases for load_env function."""

    @patch("datarobot_genai.drmcp.test_utils.utils.load_dotenv")
    def test_load_env_calls_load_dotenv(self, mock_load_dotenv) -> None:
        """Test that load_env calls load_dotenv with correct parameters."""
        load_env()

        mock_load_dotenv.assert_called_once_with(dotenv_path=".env", verbose=True, override=True)
