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
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest
import pytest_asyncio

from datarobot_genai.drmcp.core.dynamic_prompts.controllers import delete_registered_prompt_template
from datarobot_genai.drmcp.core.dynamic_prompts.controllers import get_registered_prompt_templates
from datarobot_genai.drmcp.core.mcp_instance import TaggedFastMCP


@pytest_asyncio.fixture
async def mcp_server() -> AsyncIterator[TaggedFastMCP]:
    """Create a separate MCP instance for testing."""
    test_mcp = TaggedFastMCP()

    # Patch the mcp import in controllers, register modules, and mcp_instance
    with (
        patch("datarobot_genai.drmcp.core.dynamic_prompts.controllers.mcp", test_mcp),
        patch("datarobot_genai.drmcp.core.mcp_instance.mcp", test_mcp),
    ):
        yield test_mcp


class TestPromptTemplatesListing:
    """Tests for prompt templates listing functionality."""

    @pytest.mark.asyncio
    async def test_get_registered_prompt_templates(self, mcp_server: TaggedFastMCP) -> None:
        """Test listing registered prompt templates when data exist."""
        # Setup - add test prompts directly to MCP
        await mcp_server.set_prompt_mapping("pt1", "ptv1.1", "abc")
        await mcp_server.set_prompt_mapping("pt1", "ptv1.2", "abc2")
        await mcp_server.set_prompt_mapping("pt2", "ptv3", "def")

        result = await get_registered_prompt_templates()

        expected_mappings = {
            "pt1": ("ptv1.2", "abc2"),
            "pt2": ("ptv3", "def"),
        }
        assert result == expected_mappings

        # Verify MCP internal state consistency
        internal_mappings = await mcp_server.get_prompt_mapping()
        assert internal_mappings == expected_mappings

    @pytest.mark.asyncio
    async def test_get_registered_prompt_templates_when_empty(self, mcp_server: TaggedFastMCP):
        """Test listing registered prompt templates when no data exist."""
        result = await get_registered_prompt_templates()

        assert result == {}

        # Verify MCP internal state consistency
        internal_mappings = await mcp_server.get_prompt_mapping()
        assert internal_mappings == {}


class TestPromptTemplatesDeletion:
    """Tests for prompt templates deletion functionality."""

    @pytest.mark.asyncio
    async def test_delete_registered_prompt_template(self, mcp_server: TaggedFastMCP) -> None:
        """Test delete registered prompt template."""
        # Setup - add test prompts directly to MCP
        await mcp_server.set_prompt_mapping("pt1", "ptv1.1", "abc")

        result = await delete_registered_prompt_template("pt1")

        assert result is True

        # Verify MCP internal state consistency
        internal_mappings = await mcp_server.get_prompt_mapping()
        assert internal_mappings == {}

    @pytest.mark.asyncio
    async def test_delete_not_existing_prompt_template(self, mcp_server: TaggedFastMCP):
        """Test delete not existing prompt template."""
        # Setup - add test prompts directly to MCP
        await mcp_server.set_prompt_mapping("pt1", "ptv1.1", "abc")

        result = await delete_registered_prompt_template("pt2")

        assert result is False

        # Verify MCP internal state consistency
        internal_mappings = await mcp_server.get_prompt_mapping()
        assert internal_mappings == {
            "pt1": ("ptv1.1", "abc"),
        }
