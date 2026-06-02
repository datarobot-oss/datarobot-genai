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
from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import datarobot as dr
import pytest

from datarobot_genai.drmcp.core.dynamic_prompts.controllers import delete_registered_prompt_template
from datarobot_genai.drmcp.core.dynamic_prompts.controllers import (
    refresh_registered_prompt_template,
)
from datarobot_genai.drmcp.core.dynamic_prompts.controllers import (
    register_prompt_from_prompt_template_id_and_version,
)
from datarobot_genai.drmcp.core.exceptions import DynamicPromptRegistrationError
from datarobot_genai.drmcp.core.mcp_instance import DataRobotMCP


@pytest.fixture
def mcp_server() -> Iterator[DataRobotMCP]:
    """Create a separate MCP instance for testing."""
    test_mcp = DataRobotMCP()

    # Patch the mcp import in controllers, register modules, and mcp_instance
    with (
        patch("datarobot_genai.drmcp.core.dynamic_prompts.controllers.mcp", test_mcp),
        patch("datarobot_genai.drmcp.core.mcp_instance.mcp", test_mcp),
    ):
        yield test_mcp


@pytest.fixture
def dr_lib_mock() -> Iterator[None]:
    prompt_template = dr.genai.PromptTemplate(
        id="pt1", name="pt1 name", description="pt1 description"
    )
    prompt_template_version = dr.genai.PromptTemplateVersion(
        id="ptv1.1", prompt_template_id="pt1", version=1, prompt_text="Text 1", variables=[]
    )

    with (
        patch(
            "datarobot_genai.drmcp.core.dynamic_prompts.controllers.get_datarobot_prompt_template",
            Mock(return_value=prompt_template),
        ),
        patch(
            "datarobot_genai.drmcp.core.dynamic_prompts.controllers.get_datarobot_prompt_template_version",
            Mock(return_value=prompt_template_version),
        ),
    ):
        yield


@pytest.fixture
def dr_lib_mock_empty() -> Iterator[None]:
    with (
        patch(
            "datarobot_genai.drmcp.core.dynamic_prompts.controllers.get_datarobot_prompt_template",
            Mock(return_value=None),
        ),
    ):
        yield


@pytest.fixture
def dr_lib_mock_for_refresh() -> Iterator[None]:
    prompt_template_1 = dr.genai.PromptTemplate(
        id="pt1", name="pt1 name", description="pt1 description"
    )
    prompt_template_version_1 = dr.genai.PromptTemplateVersion(
        id="ptv1.2",
        prompt_template_id=prompt_template_1.id,
        version=2,
        prompt_text="Text 1 (updated)",
        variables=[],
    )
    prompt_template_3 = dr.genai.PromptTemplate(
        id="pt3", name="pt3 name", description="pt3 description"
    )
    prompt_template_version_3 = dr.genai.PromptTemplateVersion(
        id="ptv3.1",
        prompt_template_id=prompt_template_3.id,
        version=3,
        prompt_text="Text 3",
        variables=[],
    )

    with (
        patch(
            "datarobot_genai.drmcp.core.dynamic_prompts.controllers.get_datarobot_prompt_templates",
            Mock(return_value=[prompt_template_1, prompt_template_3]),
        ),
        patch(
            "datarobot_genai.drmcp.core.dynamic_prompts.controllers.get_datarobot_prompt_template_versions",
            Mock(
                return_value={
                    prompt_template_1.id: [prompt_template_version_1],
                    prompt_template_3.id: [prompt_template_version_3],
                }
            ),
        ),
    ):
        yield


@pytest.fixture
def mock_module_under_test() -> str:
    return "datarobot_genai.drmcp.core.dynamic_prompts.controllers"


class TestPromptTemplatesAdd:
    """Tests for prompt templates add/update functionality."""

    @pytest.fixture
    def mock_get_datarobot_prompt_template(self, mock_module_under_test: str) -> Iterator[Mock]:
        with patch(f"{mock_module_under_test}.get_datarobot_prompt_template") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_datarobot_prompt_template_version(
        self, mock_module_under_test: str
    ) -> Iterator[Mock]:
        with patch(f"{mock_module_under_test}.get_datarobot_prompt_template_version") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_register_prompt_from_datarobot_prompt_management(
        self,
        mock_module_under_test: str,
    ) -> Iterator[AsyncMock]:
        with patch(
            f"{mock_module_under_test}.register_prompt_from_datarobot_prompt_management",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_is_mcp_tools_gallery_support_enabled",
        "mock_lineage_manager_init",
        "mock_sync_mcp_prompts",
    )
    async def test_add_prompt_templates(self, mcp_server: DataRobotMCP, dr_lib_mock: None) -> None:
        """Test add prompt template."""
        # Check if there's no data at the beginning
        existing_prompts = await mcp_server.get_prompt_mapping()
        assert existing_prompts == {}

        await register_prompt_from_prompt_template_id_and_version(
            prompt_template_id="pt1", prompt_template_version_id="ptv1.1"
        )

        # Verify MCP internal state consistency
        internal_mappings = await mcp_server.get_prompt_mapping()
        assert internal_mappings == {
            "pt1": ("ptv1.1", "pt1 name"),
        }

    @pytest.mark.asyncio
    async def test_add_prompt_template_when_does_not_exist(
        self, mcp_server: DataRobotMCP, dr_lib_mock_empty: None
    ) -> None:
        """Test add prompt template when does not exist in DR."""
        # Check if there's no data at the beginning
        existing_prompts = await mcp_server.get_prompt_mapping()
        assert existing_prompts == {}

        with pytest.raises(DynamicPromptRegistrationError):
            await register_prompt_from_prompt_template_id_and_version(
                prompt_template_id="pt_not_existing", prompt_template_version_id="ptv1.1"
            )

        # Verify MCP internal state consistency
        internal_mappings = await mcp_server.get_prompt_mapping()
        assert internal_mappings == {}

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_is_mcp_tools_gallery_support_enabled",
        "mock_lineage_manager_init",
        "mock_sync_mcp_prompts",
    )
    async def test_update_prompt_template(
        self, mcp_server: DataRobotMCP, dr_lib_mock: None
    ) -> None:
        """Test update prompt template."""
        # Check if there's no data at the beginning
        await mcp_server.set_prompt_mapping("pt1", "ptv1.0", "dummy v0")
        existing_prompts = await mcp_server.get_prompt_mapping()
        assert existing_prompts == {"pt1": ("ptv1.0", "dummy v0")}

        await register_prompt_from_prompt_template_id_and_version(
            prompt_template_id="pt1", prompt_template_version_id="ptv1.1"
        )

        # Verify MCP internal state consistency
        internal_mappings = await mcp_server.get_prompt_mapping()
        assert internal_mappings == {"pt1": ("ptv1.1", "pt1 name")}

    @pytest.mark.asyncio
    async def test_sync_mcp_metadata_after_registering_with_prompt_template_id(
        self,
        mock_is_mcp_tools_gallery_support_enabled: Mock,
        mock_get_datarobot_prompt_template: Mock,
        mock_lineage_manager_init: Mock,
        mock_sync_mcp_prompts: Mock,
        mock_register_prompt_from_datarobot_prompt_management: AsyncMock,
        mcp_server: DataRobotMCP,
    ) -> None:
        prompt_template_id = Mock()
        await register_prompt_from_prompt_template_id_and_version(prompt_template_id, None)

        mock_prompt_template = mock_get_datarobot_prompt_template.return_value
        mock_register_prompt_from_datarobot_prompt_management.assert_called_once_with(
            prompt_template=mock_prompt_template,
        )
        mock_is_mcp_tools_gallery_support_enabled.assert_called_once_with()
        mock_lineage_manager_init.assert_called_once_with(mcp_server)
        mock_sync_mcp_prompts.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_sync_mcp_metadata_after_registering_with_prompt_template_and_version_ids(
        self,
        mock_is_mcp_tools_gallery_support_enabled: Mock,
        mock_get_datarobot_prompt_template: Mock,
        mock_get_datarobot_prompt_template_version: Mock,
        mock_lineage_manager_init: Mock,
        mock_sync_mcp_prompts: Mock,
        mock_register_prompt_from_datarobot_prompt_management: Mock,
        mcp_server: DataRobotMCP,
    ) -> None:
        prompt_template_id = Mock()
        prompt_template_version_id = Mock()
        await register_prompt_from_prompt_template_id_and_version(
            prompt_template_id,
            prompt_template_version_id,
        )

        mock_get_datarobot_prompt_template.assert_called_once_with(prompt_template_id)
        mock_prompt_template = mock_get_datarobot_prompt_template.return_value
        mock_get_datarobot_prompt_template_version.assert_called_once_with(
            prompt_template_id, prompt_template_version_id
        )
        mock_prompt_template_version = mock_get_datarobot_prompt_template_version.return_value
        mock_register_prompt_from_datarobot_prompt_management.assert_called_once_with(
            prompt_template=mock_prompt_template,
            prompt_template_version=mock_prompt_template_version,
        )
        mock_is_mcp_tools_gallery_support_enabled.assert_called_once_with()
        mock_lineage_manager_init.assert_called_once_with(mcp_server)
        mock_sync_mcp_prompts.assert_called_once_with()


class TestPromptTemplatesListing:
    """Tests for prompt templates listing functionality."""

    @pytest.mark.asyncio
    async def test_get_registered_prompt_templates(self, mcp_server: DataRobotMCP) -> None:
        """Test listing registered prompt templates when data exist."""
        # Setup - add test prompts directly to MCP
        await mcp_server.set_prompt_mapping("pt1", "ptv1.1", "abc")
        await mcp_server.set_prompt_mapping("pt1", "ptv1.2", "abc2")
        await mcp_server.set_prompt_mapping("pt2", "ptv3", "def")

        expected_mappings = {
            "pt1": ("ptv1.2", "abc2"),
            "pt2": ("ptv3", "def"),
        }

        # Verify MCP internal state consistency
        internal_mappings = await mcp_server.get_prompt_mapping()
        assert internal_mappings == expected_mappings

    @pytest.mark.asyncio
    async def test_get_registered_prompt_templates_when_empty(self, mcp_server: DataRobotMCP):
        """Test listing registered prompt templates when no data exist."""
        # Verify MCP internal state consistency
        internal_mappings = await mcp_server.get_prompt_mapping()
        assert internal_mappings == {}


class TestPromptTemplatesDeletion:
    """Tests for prompt templates deletion functionality."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_is_mcp_tools_gallery_support_enabled",
        "mock_lineage_manager_init",
        "mock_sync_mcp_prompts",
    )
    async def test_delete_registered_prompt_template(self, mcp_server: DataRobotMCP) -> None:
        """Test delete registered prompt template."""
        # Setup - add test prompts directly to MCP
        await mcp_server.set_prompt_mapping("pt1", "ptv1.1", "abc")

        result = await delete_registered_prompt_template("pt1")

        assert result is True

        # Verify MCP internal state consistency
        internal_mappings = await mcp_server.get_prompt_mapping()
        assert internal_mappings == {}

    @pytest.mark.asyncio
    async def test_delete_not_existing_prompt_template(self, mcp_server: DataRobotMCP):
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

    @pytest.mark.asyncio
    async def test_sync_mcp_metadata_after_deletion(
        self,
        mock_is_mcp_tools_gallery_support_enabled: Mock,
        mock_lineage_manager_init: Mock,
        mock_sync_mcp_prompts: Mock,
        mcp_server: DataRobotMCP,
    ) -> None:
        prompt_id = Mock()
        await mcp_server.set_prompt_mapping(prompt_id, Mock(), Mock())

        await delete_registered_prompt_template(prompt_id)

        mock_is_mcp_tools_gallery_support_enabled.assert_called_once_with()
        mock_lineage_manager_init.assert_called_once_with(mcp_server)
        mock_sync_mcp_prompts.assert_called_once_with()


class TestPromptTemplatesRefresh:
    """Tests for prompt templates refresh functionality."""

    @pytest.fixture
    def mock_get_datarobot_prompt_templates(self, mock_module_under_test: str) -> Iterator[Mock]:
        with patch(f"{mock_module_under_test}.get_datarobot_prompt_templates") as mock_func:
            mock_func.return_value = []
            yield mock_func

    @pytest.fixture
    def mock_get_datarobot_prompt_template_versions(
        self, mock_module_under_test: str
    ) -> Iterator[Mock]:
        with patch(f"{mock_module_under_test}.get_datarobot_prompt_template_versions") as mock_func:
            mock_func.return_value = {}
            yield mock_func

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_is_mcp_tools_gallery_support_enabled",
        "mock_lineage_manager_init",
        "mock_sync_mcp_prompts",
    )
    async def test_refresh_prompt_templates(
        self, mcp_server: DataRobotMCP, dr_lib_mock_for_refresh: None
    ) -> None:
        """
        Test refresh prompt templates.

        In this test:
        - 2 prompt template already registered in MCP
        - 1 prompt template will be deleted in prompt templates API
        - 1 prompt template will be updated in prompt templates API
        - 1 prompt template will be added in prompt templates API
        """
        # Setup - add test prompts directly to MCP
        await mcp_server.set_prompt_mapping("pt1", "ptv1.1", "abc")
        await mcp_server.set_prompt_mapping("pt2", "ptv2.1", "def")

        await refresh_registered_prompt_template()

        # Verify MCP internal state consistency
        internal_mappings = await mcp_server.get_prompt_mapping()
        assert internal_mappings == {
            "pt1": ("ptv1.2", "pt1 name"),  # Updated v1.1 -> v1.2
            "pt3": ("ptv3.1", "pt3 name"),  # New pt3
            # And pt2 deleted
        }

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_get_datarobot_prompt_templates",
        "mock_get_datarobot_prompt_template_versions",
    )
    async def test_sync_mcp_metadata_after_refresh_prompts(
        self,
        mock_is_mcp_tools_gallery_support_enabled: Mock,
        mock_lineage_manager_init: Mock,
        mock_sync_mcp_prompts: Mock,
        mcp_server: DataRobotMCP,
    ) -> None:
        await refresh_registered_prompt_template()

        mock_is_mcp_tools_gallery_support_enabled.assert_called_once_with()
        mock_lineage_manager_init.assert_called_once_with(mcp_server)
        mock_sync_mcp_prompts.assert_called_once_with()
