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

from unittest.mock import Mock
from unittest.mock import mock_open
from unittest.mock import patch

from fastmcp.tools import Tool
from mcp.types import Prompt as MCPPrompt
from mcp.types import Resource as MCPResource
from mcp.types import Tool as MCPTool

from datarobot_genai.drmcp.core.utils import filter_tools_by_tags
from datarobot_genai.drmcp.core.utils import get_prompt_tags
from datarobot_genai.drmcp.core.utils import get_resource_tags
from datarobot_genai.drmcp.core.utils import get_tool_tags
from datarobot_genai.drmcp.test_utils.clients.base import LLMResponse
from datarobot_genai.drmcp.test_utils.clients.base import ToolCall
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


class TestGetToolTags:
    """Test cases for get_tool_tags function."""

    def test_get_tool_tags_fastmcp_tool_with_tags(self):
        """Test get_tool_tags with FastMCP Tool that has tags."""
        # Create a mock FastMCP Tool with tags
        tool = Mock(spec=Tool)
        tool.tags = {"tag1", "tag2", "tag3"}

        result = get_tool_tags(tool)

        assert result == {"tag1", "tag2", "tag3"}

    def test_get_tool_tags_fastmcp_tool_without_tags(self):
        """Test get_tool_tags with FastMCP Tool that has no tags attribute."""
        # Create a mock FastMCP Tool without tags
        tool = Mock(spec=Tool)
        del tool.tags  # Remove tags attribute

        result = get_tool_tags(tool)

        assert result == set()

    def test_get_tool_tags_fastmcp_tool_with_none_tags(self):
        """Test get_tool_tags with FastMCP Tool that has tags=None."""
        # Create a mock FastMCP Tool with tags=None
        tool = Mock(spec=Tool)
        tool.tags = None

        result = get_tool_tags(tool)

        assert result == set()

    def test_get_tool_tags_fastmcp_tool_with_empty_tags(self):
        """Test get_tool_tags with FastMCP Tool that has empty tags set."""
        # Create a mock FastMCP Tool with empty tags
        tool = Mock(spec=Tool)
        tool.tags = set()

        result = get_tool_tags(tool)

        assert result == set()

    def test_get_tool_tags_mcptool_with_tags(self):
        """Test get_tool_tags with MCPTool that has tags in meta._fastmcp.tags."""
        # Create a mock MCPTool with tags in meta
        tool = Mock(spec=MCPTool)
        tool.meta = {"_fastmcp": {"tags": ["tag1", "tag2", "tag3"]}}

        result = get_tool_tags(tool)

        assert result == {"tag1", "tag2", "tag3"}

    def test_get_tool_tags_mcptool_with_empty_tags_list(self):
        """Test get_tool_tags with MCPTool that has empty tags list."""
        # Create a mock MCPTool with empty tags list
        tool = Mock(spec=MCPTool)
        tool.meta = {"_fastmcp": {"tags": []}}

        result = get_tool_tags(tool)

        assert result == set()

    def test_get_tool_tags_mcptool_without_meta(self):
        """Test get_tool_tags with MCPTool that has no meta."""
        # Create a mock MCPTool without meta
        tool = Mock(spec=MCPTool)
        tool.meta = None

        result = get_tool_tags(tool)

        assert result == set()

    def test_get_tool_tags_mcptool_with_empty_meta(self):
        """Test get_tool_tags with MCPTool that has empty meta dict."""
        # Create a mock MCPTool with empty meta
        tool = Mock(spec=MCPTool)
        tool.meta = {}

        result = get_tool_tags(tool)

        assert result == set()

    def test_get_tool_tags_mcptool_with_meta_but_no_fastmcp(self):
        """Test get_tool_tags with MCPTool that has meta but no _fastmcp key."""
        # Create a mock MCPTool with meta but no _fastmcp
        tool = Mock(spec=MCPTool)
        tool.meta = {"other_key": "value"}

        result = get_tool_tags(tool)

        assert result == set()

    def test_get_tool_tags_mcptool_with_fastmcp_but_no_tags(self):
        """Test get_tool_tags with MCPTool that has _fastmcp but no tags."""
        # Create a mock MCPTool with _fastmcp but no tags
        tool = Mock(spec=MCPTool)
        tool.meta = {"_fastmcp": {"other_key": "value"}}

        result = get_tool_tags(tool)

        assert result == set()

    def test_get_tool_tags_mcptool_with_fastmcp_none(self):
        """Test get_tool_tags with MCPTool that has _fastmcp=None."""
        # Create a mock MCPTool with _fastmcp=None
        tool = Mock(spec=MCPTool)
        tool.meta = {"_fastmcp": None}

        result = get_tool_tags(tool)

        assert result == set()

    def test_get_tool_tags_mcptool_with_fastmcp_not_dict(self):
        """Test get_tool_tags with MCPTool that has _fastmcp as non-dict."""
        # Create a mock MCPTool with _fastmcp as string (not dict)
        tool = Mock(spec=MCPTool)
        tool.meta = {"_fastmcp": "not_a_dict"}

        result = get_tool_tags(tool)

        assert result == set()

    def test_get_tool_tags_mcptool_with_meta_not_dict(self):
        """Test get_tool_tags with MCPTool that has meta as non-dict."""
        # Create a mock MCPTool with meta as string (not dict)
        tool = Mock(spec=MCPTool)
        tool.meta = "not_a_dict"

        result = get_tool_tags(tool)

        assert result == set()


class TestFilterToolsByTags:
    """Test cases for filter_tools_by_tags function."""

    def test_filter_tools_by_tags_no_tags_provided(self):
        """Test filter_tools_by_tags when no tags are provided (returns all tools)."""
        # Create mock tools
        tool1 = Mock(spec=Tool)
        tool1.tags = {"tag1"}
        tool2 = Mock(spec=Tool)
        tool2.tags = {"tag2"}

        tools = [tool1, tool2]

        result = filter_tools_by_tags(tools=tools, tags=None)

        assert result == tools
        assert len(result) == 2

    def test_filter_tools_by_tags_empty_tags_list(self):
        """Test filter_tools_by_tags when empty tags list is provided (returns all tools)."""
        # Create mock tools
        tool1 = Mock(spec=Tool)
        tool1.tags = {"tag1"}
        tool2 = Mock(spec=Tool)
        tool2.tags = {"tag2"}

        tools = [tool1, tool2]

        result = filter_tools_by_tags(tools=tools, tags=[])

        assert result == tools
        assert len(result) == 2

    def test_filter_tools_by_tags_match_any_single_tag(self):
        """Test filter_tools_by_tags with match_all=False (match any tag)."""
        # Create mock tools with different tags
        tool1 = Mock(spec=Tool)
        tool1.tags = {"tag1", "tag2"}
        tool2 = Mock(spec=Tool)
        tool2.tags = {"tag3"}
        tool3 = Mock(spec=Tool)
        tool3.tags = {"tag2", "tag4"}

        tools = [tool1, tool2, tool3]

        # Filter for tools that have tag1 or tag2
        result = filter_tools_by_tags(tools=tools, tags=["tag1", "tag2"], match_all=False)

        # Should return tool1 (has tag1 and tag2) and tool3 (has tag2)
        assert len(result) == 2
        assert tool1 in result
        assert tool3 in result
        assert tool2 not in result

    def test_filter_tools_by_tags_match_all_tags(self):
        """Test filter_tools_by_tags with match_all=True (must have all tags)."""
        # Create mock tools with different tags
        tool1 = Mock(spec=Tool)
        tool1.tags = {"tag1", "tag2", "tag3"}
        tool2 = Mock(spec=Tool)
        tool2.tags = {"tag1", "tag2"}
        tool3 = Mock(spec=Tool)
        tool3.tags = {"tag2", "tag3"}

        tools = [tool1, tool2, tool3]

        # Filter for tools that have both tag1 and tag2
        result = filter_tools_by_tags(tools=tools, tags=["tag1", "tag2"], match_all=True)

        # Should return tool1 (has both) and tool2 (has both)
        assert len(result) == 2
        assert tool1 in result
        assert tool2 in result
        assert tool3 not in result

    def test_filter_tools_by_tags_match_all_no_match(self):
        """Test filter_tools_by_tags with match_all=True when no tool has all tags."""
        # Create mock tools
        tool1 = Mock(spec=Tool)
        tool1.tags = {"tag1"}
        tool2 = Mock(spec=Tool)
        tool2.tags = {"tag2"}

        tools = [tool1, tool2]

        # Filter for tools that have both tag1 and tag2
        result = filter_tools_by_tags(tools=tools, tags=["tag1", "tag2"], match_all=True)

        # Should return empty list
        assert result == []
        assert len(result) == 0

    def test_filter_tools_by_tags_empty_tools_list(self):
        """Test filter_tools_by_tags with empty tools list."""
        tools = []

        result = filter_tools_by_tags(tools=tools, tags=["tag1"])

        assert result == []
        assert len(result) == 0

    def test_filter_tools_by_tags_tools_without_tags(self):
        """Test filter_tools_by_tags when tools have no tags."""
        # Create mock tools without tags
        tool1 = Mock(spec=Tool)
        del tool1.tags
        tool2 = Mock(spec=Tool)
        tool2.tags = set()

        tools = [tool1, tool2]

        result = filter_tools_by_tags(tools=tools, tags=["tag1"], match_all=False)

        # Should return empty list since no tools have tags
        assert result == []
        assert len(result) == 0

    def test_filter_tools_by_tags_mixed_tool_types(self):
        """Test filter_tools_by_tags with both FastMCP Tool and MCPTool."""
        # Create FastMCP Tool
        fastmcp_tool = Mock(spec=Tool)
        fastmcp_tool.tags = {"tag1", "tag2"}

        # Create MCPTool
        mcptool = Mock(spec=MCPTool)
        mcptool.meta = {"_fastmcp": {"tags": ["tag2", "tag3"]}}

        tools = [fastmcp_tool, mcptool]

        # Filter for tools that have tag2
        result = filter_tools_by_tags(tools=tools, tags=["tag2"], match_all=False)

        # Both tools should match
        assert len(result) == 2
        assert fastmcp_tool in result
        assert mcptool in result

    def test_filter_tools_by_tags_match_all_single_tag(self):
        """Test filter_tools_by_tags with match_all=True and single tag."""
        # Create mock tools
        tool1 = Mock(spec=Tool)
        tool1.tags = {"tag1", "tag2"}
        tool2 = Mock(spec=Tool)
        tool2.tags = {"tag2"}

        tools = [tool1, tool2]

        # Filter for tools that have tag1 (single tag, match_all=True)
        result = filter_tools_by_tags(tools=tools, tags=["tag1"], match_all=True)

        # Should return tool1 only
        assert len(result) == 1
        assert tool1 in result
        assert tool2 not in result

    def test_filter_tools_by_tags_match_any_all_tools_match(self):
        """Test filter_tools_by_tags with match_all=False when all tools match."""
        # Create mock tools
        tool1 = Mock(spec=Tool)
        tool1.tags = {"tag1", "tag2"}
        tool2 = Mock(spec=Tool)
        tool2.tags = {"tag1", "tag3"}

        tools = [tool1, tool2]

        # Filter for tools that have tag1
        result = filter_tools_by_tags(tools=tools, tags=["tag1"], match_all=False)

        # Both tools should match
        assert len(result) == 2
        assert tool1 in result
        assert tool2 in result

    def test_filter_tools_by_tags_match_all_exact_match(self):
        """Test filter_tools_by_tags with match_all=True when tool has exactly the required tags."""
        # Create mock tool with exactly the required tags
        tool1 = Mock(spec=Tool)
        tool1.tags = {"tag1", "tag2"}
        tool2 = Mock(spec=Tool)
        tool2.tags = {"tag1", "tag2", "tag3"}

        tools = [tool1, tool2]

        # Filter for tools that have both tag1 and tag2
        result = filter_tools_by_tags(tools=tools, tags=["tag1", "tag2"], match_all=True)

        # Both tools should match (having all required tags)
        assert len(result) == 2
        assert tool1 in result
        assert tool2 in result


class TestGetPromptTags:
    """Test cases for get_prompt_tags function."""

    def test_get_prompt_tags_with_tags(self):
        """Test get_prompt_tags with MCPPrompt that has tags in meta._fastmcp.tags."""
        # Create a mock MCPPrompt with tags in meta
        prompt = Mock(spec=MCPPrompt)
        prompt.meta = {"_fastmcp": {"tags": ["tag1", "tag2", "tag3"]}}

        result = get_prompt_tags(prompt)

        assert result == {"tag1", "tag2", "tag3"}

    def test_get_prompt_tags_with_empty_tags_list(self):
        """Test get_prompt_tags with MCPPrompt that has empty tags list."""
        # Create a mock MCPPrompt with empty tags list
        prompt = Mock(spec=MCPPrompt)
        prompt.meta = {"_fastmcp": {"tags": []}}

        result = get_prompt_tags(prompt)

        assert result == set()

    def test_get_prompt_tags_without_meta(self):
        """Test get_prompt_tags with MCPPrompt that has no meta."""
        # Create a mock MCPPrompt without meta
        prompt = Mock(spec=MCPPrompt)
        prompt.meta = None

        result = get_prompt_tags(prompt)

        assert result == set()

    def test_get_prompt_tags_with_empty_meta(self):
        """Test get_prompt_tags with MCPPrompt that has empty meta dict."""
        # Create a mock MCPPrompt with empty meta
        prompt = Mock(spec=MCPPrompt)
        prompt.meta = {}

        result = get_prompt_tags(prompt)

        assert result == set()

    def test_get_prompt_tags_with_meta_but_no_fastmcp(self):
        """Test get_prompt_tags with MCPPrompt that has meta but no _fastmcp key."""
        # Create a mock MCPPrompt with meta but no _fastmcp
        prompt = Mock(spec=MCPPrompt)
        prompt.meta = {"other_key": "value"}

        result = get_prompt_tags(prompt)

        assert result == set()

    def test_get_prompt_tags_with_fastmcp_but_no_tags(self):
        """Test get_prompt_tags with MCPPrompt that has _fastmcp but no tags."""
        # Create a mock MCPPrompt with _fastmcp but no tags
        prompt = Mock(spec=MCPPrompt)
        prompt.meta = {"_fastmcp": {"other_key": "value"}}

        result = get_prompt_tags(prompt)

        assert result == set()

    def test_get_prompt_tags_with_fastmcp_none(self):
        """Test get_prompt_tags with MCPPrompt that has _fastmcp=None."""
        # Create a mock MCPPrompt with _fastmcp=None
        prompt = Mock(spec=MCPPrompt)
        prompt.meta = {"_fastmcp": None}

        result = get_prompt_tags(prompt)

        assert result == set()

    def test_get_prompt_tags_with_fastmcp_not_dict(self):
        """Test get_prompt_tags with MCPPrompt that has _fastmcp as non-dict."""
        # Create a mock MCPPrompt with _fastmcp as string (not dict)
        prompt = Mock(spec=MCPPrompt)
        prompt.meta = {"_fastmcp": "not_a_dict"}

        result = get_prompt_tags(prompt)

        assert result == set()

    def test_get_prompt_tags_with_meta_not_dict(self):
        """Test get_prompt_tags with MCPPrompt that has meta as non-dict."""
        # Create a mock MCPPrompt with meta as string (not dict)
        prompt = Mock(spec=MCPPrompt)
        prompt.meta = "not_a_dict"

        result = get_prompt_tags(prompt)

        assert result == set()


class TestGetResourceTags:
    """Test cases for get_resource_tags function."""

    def test_get_resource_tags_with_tags(self):
        """Test get_resource_tags with MCPResource that has tags in meta._fastmcp.tags."""
        # Create a mock MCPResource with tags in meta
        resource = Mock(spec=MCPResource)
        resource.meta = {"_fastmcp": {"tags": ["tag1", "tag2", "tag3"]}}

        result = get_resource_tags(resource)

        assert result == {"tag1", "tag2", "tag3"}

    def test_get_resource_tags_with_empty_tags_list(self):
        """Test get_resource_tags with MCPResource that has empty tags list."""
        # Create a mock MCPResource with empty tags list
        resource = Mock(spec=MCPResource)
        resource.meta = {"_fastmcp": {"tags": []}}

        result = get_resource_tags(resource)

        assert result == set()

    def test_get_resource_tags_without_meta(self):
        """Test get_resource_tags with MCPResource that has no meta."""
        # Create a mock MCPResource without meta
        resource = Mock(spec=MCPResource)
        resource.meta = None

        result = get_resource_tags(resource)

        assert result == set()

    def test_get_resource_tags_with_empty_meta(self):
        """Test get_resource_tags with MCPResource that has empty meta dict."""
        # Create a mock MCPResource with empty meta
        resource = Mock(spec=MCPResource)
        resource.meta = {}

        result = get_resource_tags(resource)

        assert result == set()

    def test_get_resource_tags_with_meta_but_no_fastmcp(self):
        """Test get_resource_tags with MCPResource that has meta but no _fastmcp key."""
        # Create a mock MCPResource with meta but no _fastmcp
        resource = Mock(spec=MCPResource)
        resource.meta = {"other_key": "value"}

        result = get_resource_tags(resource)

        assert result == set()

    def test_get_resource_tags_with_fastmcp_but_no_tags(self):
        """Test get_resource_tags with MCPResource that has _fastmcp but no tags."""
        # Create a mock MCPResource with _fastmcp but no tags
        resource = Mock(spec=MCPResource)
        resource.meta = {"_fastmcp": {"other_key": "value"}}

        result = get_resource_tags(resource)

        assert result == set()

    def test_get_resource_tags_with_fastmcp_none(self):
        """Test get_resource_tags with MCPResource that has _fastmcp=None."""
        # Create a mock MCPResource with _fastmcp=None
        resource = Mock(spec=MCPResource)
        resource.meta = {"_fastmcp": None}

        result = get_resource_tags(resource)

        assert result == set()

    def test_get_resource_tags_with_fastmcp_not_dict(self):
        """Test get_resource_tags with MCPResource that has _fastmcp as non-dict."""
        # Create a mock MCPResource with _fastmcp as string (not dict)
        resource = Mock(spec=MCPResource)
        resource.meta = {"_fastmcp": "not_a_dict"}

        result = get_resource_tags(resource)

        assert result == set()

    def test_get_resource_tags_with_meta_not_dict(self):
        """Test get_resource_tags with MCPResource that has meta as non-dict."""
        # Create a mock MCPResource with meta as string (not dict)
        resource = Mock(spec=MCPResource)
        resource.meta = "not_a_dict"

        result = get_resource_tags(resource)

        assert result == set()
