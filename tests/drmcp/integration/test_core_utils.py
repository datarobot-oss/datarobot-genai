# Copyright 2026 DataRobot, Inc.
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

"""Integration tests for core/utils.py module."""

import pytest
from mcp.types import ListToolsResult
from mcp.types import Tool as MCPTool

from datarobot_genai.drmcp.core.utils import filter_tools_by_tags
from datarobot_genai.drmcp.core.utils import get_tool_tags
from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session


@pytest.mark.asyncio
class TestCoreUtilsIntegration:
    """Integration tests for core/utils.py functions."""

    async def test_get_tool_tags_with_mcptool_from_session(self) -> None:
        """Test get_tool_tags with real MCPTool objects from MCP session."""
        async with integration_test_mcp_session() as session:
            # Get tools from the session
            tools_result: ListToolsResult = await session.list_tools()
            tools = tools_result.tools

            # Test get_tool_tags on all tools from the session
            for tool in tools:
                # All tools from session should be MCPTool instances
                assert isinstance(tool, MCPTool)

                # get_tool_tags should handle MCPTool objects gracefully
                tags = get_tool_tags(tool)

                # Tags should be a set (empty or with values)
                assert isinstance(tags, set)

                # If tool has meta with _fastmcp.tags, verify extraction
                if tool.meta and isinstance(tool.meta, dict):
                    fastmcp_meta = tool.meta.get("_fastmcp")
                    if fastmcp_meta and isinstance(fastmcp_meta, dict):
                        meta_tags = fastmcp_meta.get("tags")
                        if meta_tags:
                            # Tags should match what's in meta
                            assert tags == set(meta_tags)
                        else:
                            # No tags in meta, should return empty set
                            assert tags == set()
                    else:
                        # No _fastmcp in meta, should return empty set
                        assert tags == set()
                else:
                    # No meta or meta is not dict, should return empty set
                    assert tags == set()

    async def test_get_tool_tags_with_mcptool_various_meta_structures(self) -> None:
        """Test get_tool_tags handles various meta structures in MCPTool objects."""
        async with integration_test_mcp_session() as session:
            # Get tools from the session
            tools_result: ListToolsResult = await session.list_tools()
            tools = tools_result.tools

            # Test get_tool_tags on all tools from the session
            # This verifies the function handles real MCPTool objects correctly
            for tool in tools:
                assert isinstance(tool, MCPTool)

                # get_tool_tags should handle any MCPTool structure gracefully
                tags = get_tool_tags(tool)
                assert isinstance(tags, set)

                # Verify tags match what's in meta if present
                if tool.meta and isinstance(tool.meta, dict):
                    fastmcp_meta = tool.meta.get("_fastmcp")
                    if fastmcp_meta and isinstance(fastmcp_meta, dict):
                        meta_tags = fastmcp_meta.get("tags")
                        if meta_tags:
                            assert tags == set(meta_tags)
                        else:
                            assert tags == set()
                    else:
                        assert tags == set()
                else:
                    assert tags == set()

    async def test_filter_tools_by_tags_with_session_tools(self) -> None:
        """Test filter_tools_by_tags with real tools from MCP session."""
        async with integration_test_mcp_session() as session:
            # Get tools from the session
            tools_result: ListToolsResult = await session.list_tools()
            tools = list(tools_result.tools)

            # Test filtering with no tags (should return all tools)
            filtered = filter_tools_by_tags(tools=tools, tags=None)
            assert len(filtered) == len(tools)

            # Test filtering with empty tags list (should return all tools)
            filtered = filter_tools_by_tags(tools=tools, tags=[])
            assert len(filtered) == len(tools)

            # Test filtering with a tag that likely doesn't exist
            filtered = filter_tools_by_tags(tools=tools, tags=["nonexistent_tag"])
            # Should return empty list or tools that happen to have this tag
            assert isinstance(filtered, list)

    async def test_filter_tools_by_tags_with_session_tools_comprehensive(self) -> None:
        """Test filter_tools_by_tags comprehensively with real tools from MCP session."""
        async with integration_test_mcp_session() as session:
            # Get tools from the session
            tools_result: ListToolsResult = await session.list_tools()
            tools = list(tools_result.tools)

            # Test filtering with no tags (should return all tools)
            filtered = filter_tools_by_tags(tools=tools, tags=None)
            assert len(filtered) == len(tools)
            assert filtered == tools

            # Test filtering with empty tags list (should return all tools)
            filtered = filter_tools_by_tags(tools=tools, tags=[])
            assert len(filtered) == len(tools)

            # Test filtering with a tag that likely doesn't exist
            filtered = filter_tools_by_tags(tools=tools, tags=["nonexistent_tag_xyz"])
            # Should return empty list or tools that happen to have this tag
            assert isinstance(filtered, list)

            # Test filtering with match_all=True
            filtered = filter_tools_by_tags(
                tools=tools, tags=["nonexistent_tag_xyz"], match_all=True
            )
            assert isinstance(filtered, list)

            # Test that filtering preserves tool order and doesn't modify tools
            original_tool_names = [t.name for t in tools]
            filtered = filter_tools_by_tags(tools=tools, tags=None)
            assert [t.name for t in filtered] == original_tool_names

    async def test_get_tool_tags_and_filter_integration(self) -> None:
        """Integration test combining get_tool_tags and filter_tools_by_tags."""
        async with integration_test_mcp_session() as session:
            # Get tools from the session
            tools_result: ListToolsResult = await session.list_tools()
            tools = list(tools_result.tools)

            # Verify get_tool_tags works on all tools
            tool_tags_map = {}
            for tool in tools:
                tags = get_tool_tags(tool)
                assert isinstance(tags, set)
                tool_tags_map[tool.name] = tags

            # Test filtering: if any tools have tags, test filtering by those tags
            tools_with_tags = [tool for tool in tools if get_tool_tags(tool)]

            if tools_with_tags:
                # Get tags from the first tool that has tags
                sample_tags = get_tool_tags(tools_with_tags[0])
                if sample_tags:
                    # Filter by one of the tags from the sample tool
                    test_tag = list(sample_tags)[0]
                    filtered = filter_tools_by_tags(tools=tools, tags=[test_tag], match_all=False)
                    # At least the sample tool should be in the filtered results
                    assert len(filtered) >= 1
                    assert tools_with_tags[0] in filtered

                    # Verify all filtered tools have the test tag
                    for tool in filtered:
                        tags = get_tool_tags(tool)
                        assert test_tag in tags

            # Test that get_tool_tags and filter_tools_by_tags work together
            # Filter tools that have any tags
            tools_with_any_tags = [tool for tool in tools if get_tool_tags(tool)]

            # Verify consistency: tools returned by filter should match those with tags
            # when filtering with a tag they have
            if tools_with_any_tags:
                sample_tool = tools_with_any_tags[0]
                sample_tags = get_tool_tags(sample_tool)
                if sample_tags:
                    test_tag = list(sample_tags)[0]
                    filtered = filter_tools_by_tags(tools=tools, tags=[test_tag], match_all=False)
                    # Verify the sample tool is in filtered results
                    assert sample_tool in filtered
                    # Verify all filtered tools have the tag
                    for tool in filtered:
                        assert test_tag in get_tool_tags(tool)
