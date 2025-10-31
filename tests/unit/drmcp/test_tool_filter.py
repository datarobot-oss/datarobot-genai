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

from unittest.mock import Mock
from unittest.mock import patch

from datarobot_genai.drmcp.core.tool_filter import filter_tools_by_tags
from datarobot_genai.drmcp.core.tool_filter import get_tool_tags
from datarobot_genai.drmcp.core.tool_filter import get_tools_by_tag
from datarobot_genai.drmcp.core.tool_filter import list_all_tags


class TestToolFilterAdditional:
    """Additional test cases for tool filter functions."""

    def test_filter_tools_by_tags_no_tags_returns_all_tools(self):
        """Test that filter_tools_by_tags with no tags returns all tools."""
        tool1 = Mock()
        tool2 = Mock()
        tools = [tool1, tool2]

        result = filter_tools_by_tags(tools, tags=None)
        assert result == tools

    def test_filter_tools_by_tags_empty_tags_returns_all_tools(self):
        """Test that filter_tools_by_tags with empty tags returns all tools."""
        tool1 = Mock()
        tool2 = Mock()
        tools = [tool1, tool2]

        result = filter_tools_by_tags(tools, tags=[])
        assert result == tools

    def test_filter_tools_by_tags_tool_without_annotations_skipped(self):
        """Test that tools without annotations are skipped."""
        tool1 = Mock()
        tool1.annotations = None
        tool2 = Mock()
        tool2.annotations = Mock()
        tool2.annotations.tags = ["tag1"]
        tools = [tool1, tool2]

        result = filter_tools_by_tags(tools, tags=["tag1"])
        assert result == [tool2]

    def test_filter_tools_by_tags_tool_without_tags_skipped(self):
        """Test that tools without tags are skipped."""
        tool1 = Mock()
        tool1.annotations = Mock()
        tool1.annotations.tags = []
        tool2 = Mock()
        tool2.annotations = Mock()
        tool2.annotations.tags = ["tag1"]
        tools = [tool1, tool2]

        result = filter_tools_by_tags(tools, tags=["tag1"])
        assert result == [tool2]

    def test_filter_tools_by_tags_tool_without_tags_attribute_skipped(self):
        """Test that tools without tags attribute are skipped."""
        tool1 = Mock()
        tool1.annotations = Mock()
        # No tags attribute - getattr will return empty list
        tool1.annotations.tags = []  # Explicitly set empty list
        tool2 = Mock()
        tool2.annotations = Mock()
        tool2.annotations.tags = ["tag1"]
        tools = [tool1, tool2]

        result = filter_tools_by_tags(tools, tags=["tag1"])
        assert result == [tool2]

    def test_get_tool_tags_no_annotations_returns_empty_list(self):
        """Test that get_tool_tags with no annotations returns empty list."""
        tool = Mock()
        tool.annotations = None

        result = get_tool_tags(tool)
        assert result == []

    def test_get_tool_tags_no_tags_attribute_returns_empty_list(self):
        """Test that get_tool_tags with no tags attribute returns empty list."""
        tool = Mock()
        tool.annotations = Mock()
        # No tags attribute

        result = get_tool_tags(tool)
        assert result == []

    def test_get_tool_tags_tags_not_list_returns_empty_list(self):
        """Test that get_tool_tags with tags not being a list returns empty list."""
        tool = Mock()
        tool.annotations = Mock()
        tool.annotations.tags = "not_a_list"

        result = get_tool_tags(tool)
        assert result == []

    def test_get_tool_tags_valid_tags_returns_tags(self):
        """Test that get_tool_tags with valid tags returns the tags."""
        tool = Mock()
        tool.annotations = Mock()
        tool.annotations.tags = ["tag1", "tag2"]

        result = get_tool_tags(tool)
        assert result == ["tag1", "tag2"]

    def test_list_all_tags_empty_list_returns_empty_list(self):
        """Test that list_all_tags with empty list returns empty list."""
        result = list_all_tags([])
        assert result == []

    def test_list_all_tags_no_tags_returns_empty_list(self):
        """Test that list_all_tags with tools having no tags returns empty list."""
        tool1 = Mock()
        tool1.annotations = None
        tool2 = Mock()
        tool2.annotations = Mock()
        tool2.annotations.tags = []
        tools = [tool1, tool2]

        result = list_all_tags(tools)
        assert result == []

    def test_list_all_tags_returns_sorted_unique_tags(self):
        """Test that list_all_tags returns sorted unique tags."""
        tool1 = Mock()
        tool1.annotations = Mock()
        tool1.annotations.tags = ["tag2", "tag1"]
        tool2 = Mock()
        tool2.annotations = Mock()
        tool2.annotations.tags = ["tag1", "tag3"]
        tools = [tool1, tool2]

        result = list_all_tags(tools)
        assert result == ["tag1", "tag2", "tag3"]

    def test_get_tools_by_tag_calls_filter_tools_by_tags(self):
        """Test that get_tools_by_tag calls filter_tools_by_tags with correct parameters."""
        tool1 = Mock()
        tool2 = Mock()
        tools = [tool1, tool2]

        with patch(
            "datarobot_genai.drmcp.core.tool_filter.filter_tools_by_tags", return_value=[tool1]
        ) as mock_filter:
            result = get_tools_by_tag(tools, "tag1")

            assert result == [tool1]
            mock_filter.assert_called_once_with(tools, ["tag1"])
