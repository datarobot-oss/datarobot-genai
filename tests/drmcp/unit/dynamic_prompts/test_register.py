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

"""Tests for external prompt registration."""

from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch
from uuid import UUID

import datarobot as dr
import pytest

from datarobot_genai.drmcp.core.dynamic_prompts.register import make_prompt_function
from datarobot_genai.drmcp.core.dynamic_prompts.register import (
    register_prompts_from_datarobot_prompt_management,
)
from datarobot_genai.drmcp.core.dynamic_prompts.register import to_valid_mcp_prompt_name
from datarobot_genai.drmcp.core.mcp_instance import mcp


class TestMakePrompt:
    """Tests for make_prompt_function - happy path."""

    @pytest.mark.asyncio
    async def test_make_prompt_function(self) -> None:
        """Test making a prompt."""
        name = "dummy prompt name"
        description = "dummy prompt description"
        prompt_text = "dummy prompt text {{variable_a}} and {{variable_b}}"
        variables = [
            dr.genai.Variable(name="variable_a", description="variable_a_desc"),
            dr.genai.Variable(name="variable_b", description="variable_b_desc"),
        ]

        prompt_function = make_prompt_function(name, description, prompt_text, variables)
        prompt = await prompt_function(variable_a="variable_a_value", variable_b="variable_b_value")

        assert prompt == "dummy prompt text variable_a_value and variable_b_value"

    @pytest.mark.parametrize(
        "var_name",
        ["23124125", "class", "True", "var-name"],
    )
    def test_make_prompt_function_incorrect_variable_names(self, var_name) -> None:
        """Test making a prompt when incorrect variable names."""
        name = "dummy prompt name"
        description = "dummy prompt description"
        prompt_text = "dummy prompt text {{True}} and {{class}}"
        variables = [dr.genai.Variable(name=var_name, description=f"{var_name}_desc")]

        with pytest.raises(ValueError):
            make_prompt_function(name, description, prompt_text, variables)


class TestToValidFunctionName:
    """Tests for to_valid_function_name."""

    @pytest.mark.parametrize(
        "s,expected",
        [
            ("HeLlo W0Rld 1", "HeLlo_W0Rld_1"),
            ("1 Test Prompt", "Test_Prompt"),
            ("my-name(second)", "my_name_second"),
            ("23124125", "prompt_23124125"),
            ("class", "class_prompt"),  # Python keyword
            ("True", "True_prompt"),  # Python keyword
            ("日本語", "prompt_x65e5x672cx8a9e"),  # Full not ascii
            ("ABC 日本語", "prompt_ABC_x65e5x672cx8a9e"),  # Unicode mix with ascii
        ],
    )
    @pytest.mark.asyncio
    async def test_to_valid_function_name(self, s: str, expected: str) -> None:
        """Test converting to valid function name."""
        assert to_valid_mcp_prompt_name(s) == expected

    @pytest.mark.parametrize("s", ["$$$", "123-456-789", "---", "_123"])
    def test_to_valid_function_name_when_cannot_convert(self, s: str) -> None:
        """Test converting to valid function name."""
        with pytest.raises(ValueError):
            to_valid_mcp_prompt_name(s)


class TestRegisterPrompt:
    """Tests for register_prompt - happy path."""

    @pytest.fixture
    def module_under_test(self) -> str:
        return "datarobot_genai.drmcp.core.dynamic_prompts.register"

    @pytest.fixture
    def mock_get_datarobot_prompt_templates(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_datarobot_prompt_templates") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_datarobot_prompt_template_versions(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.get_datarobot_prompt_template_versions") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_register_prompt_from_datarobot_prompt_management(
        self, module_under_test: str
    ) -> Iterator[AsyncMock]:
        with patch(
            f"{module_under_test}.register_prompt_from_datarobot_prompt_management",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_mcp_server(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.mcp") as mock_mcp:
            yield mock_mcp

    @pytest.mark.usefixtures(
        "mock_is_mcp_tools_gallery_support_enabled",
        "mock_lineage_manager_init",
        "mock_sync_mcp_prompts",
    )
    @pytest.mark.asyncio
    async def test_register_prompt_from_datarobot_prompt_management(
        self,
        get_prompt_template_mock: None,
    ) -> None:
        """Test register prompt from dr prompt mgmt."""
        await register_prompts_from_datarobot_prompt_management()

        prompts = {prompt for prompt in await mcp.get_prompts()}

        assert "Dummy prompt name" in prompts, "`Dummy prompt name` is missing."

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_is_mcp_tools_gallery_support_enabled",
        "mock_lineage_manager_init",
        "mock_sync_mcp_prompts",
    )
    @patch("datarobot_genai.drmcp.core.dynamic_prompts.utils.uuid4")
    async def test_register_prompt_from_datarobot_prompt_management_duplicated_prompt_names(
        self,
        uuid_mock: Mock,
        get_prompt_template_duplicated_name_mock: None,
    ) -> None:
        """Test register prompt from dr prompt mgmt."""
        uuid_mock.return_value = UUID("f2bda341-4b81-48f1-8da7-e7680a7410b4")

        await register_prompts_from_datarobot_prompt_management()

        prompts = {prompt for prompt in await mcp.get_prompts()}

        assert "Dummy prompt name" in prompts, "`Dummy prompt name` is missing."
        assert "Dummy prompt name (f2bd)" in prompts, "`Dummy prompt name (f2bd)` is missing."

    @pytest.mark.asyncio
    async def test_sync_mcp_metadata_after_register_prompts(
        self,
        mock_get_datarobot_prompt_templates: Mock,
        mock_get_datarobot_prompt_template_versions: Mock,
        mock_is_mcp_tools_gallery_support_enabled: Mock,
        mock_lineage_manager_init: Mock,
        mock_sync_mcp_prompts: Mock,
        mock_mcp_server: Mock,
        mock_register_prompt_from_datarobot_prompt_management: AsyncMock,
    ) -> None:
        mock_prompt = Mock()
        mock_get_datarobot_prompt_templates.return_value = [mock_prompt]
        prompt_template_version = dr.genai.PromptTemplateVersion("adsf", version=1)
        mock_get_datarobot_prompt_template_versions.return_value = {
            mock_prompt.id: [prompt_template_version],
        }

        await register_prompts_from_datarobot_prompt_management()

        mock_get_datarobot_prompt_templates.assert_called_once_with()
        mock_get_datarobot_prompt_template_versions.assert_called_once_with(
            prompt_template_ids=[mock_prompt.id],
        )
        mock_register_prompt_from_datarobot_prompt_management.assert_called_once_with(
            mock_prompt,
            prompt_template_version,
        )
        mock_is_mcp_tools_gallery_support_enabled.assert_called_once_with()
        mock_lineage_manager_init.assert_called_once_with(mock_mcp_server)
        mock_sync_mcp_prompts.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_not_run_sync_mcp_metadata_after_no_prompt_is_registered(
        self,
        mock_get_datarobot_prompt_templates: Mock,
        mock_get_datarobot_prompt_template_versions: Mock,
        mock_is_mcp_tools_gallery_support_enabled: Mock,
        mock_lineage_manager_init: Mock,
        mock_sync_mcp_prompts: Mock,
    ) -> None:
        mock_get_datarobot_prompt_templates.return_value = []

        await register_prompts_from_datarobot_prompt_management()

        mock_get_datarobot_prompt_templates.assert_called_once_with()
        mock_get_datarobot_prompt_template_versions.assert_called_once_with(
            prompt_template_ids=[],
        )
        mock_is_mcp_tools_gallery_support_enabled.assert_not_called()
        mock_lineage_manager_init.assert_not_called()
        mock_sync_mcp_prompts.assert_not_called()
