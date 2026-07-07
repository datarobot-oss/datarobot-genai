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
from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp.prompts import Prompt

from datarobot_genai.drmcp.core.enums import DataRobotMCPPromptCategory
from datarobot_genai.drmcp.core.lineage.entities import MCPPromptMetadata
from datarobot_genai.drmcp.core.mcp_instance import DataRobotMCP
from datarobot_genai.drmcp.core.mcp_instance import (
    check_prompt_registration_status_after_it_finishes,
)
from datarobot_genai.drmcp.core.mcp_instance import register_prompt


class TestRegisterPrompt:
    @pytest.fixture
    def module_under_test(self) -> str:
        return "datarobot_genai.drmcp.core.mcp_instance"

    @pytest.fixture
    def mock_mcp_tool_callable(self) -> Iterator[Mock]:
        yield Mock(return_value=Mock())

    @pytest.fixture
    def mock_dr_mcp_extras(self, module_under_test: str) -> Iterator[Mock]:
        with patch(f"{module_under_test}.dr_mcp_extras") as mock_decorator:
            yield mock_decorator

    @pytest.fixture
    def mock_datarobot_mcp_server(
        self,
        module_under_test: str,
    ) -> Iterator[Mock]:
        with patch(f"{module_under_test}.mcp") as mock_instance:
            mock_instance.notify_prompts_changed = AsyncMock()
            mock_instance.get_prompts = AsyncMock()
            yield mock_instance

    @pytest.fixture
    def mock_get_prompt_name_no_duplicate(
        self,
        module_under_test: str,
    ) -> Iterator[AsyncMock]:
        with patch(
            f"{module_under_test}.get_prompt_name_no_duplicate",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_prompt_from_function(self) -> Iterator[Mock]:
        with patch.object(Prompt, "from_function") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_check_prompt_registration_status_after_it_finishes(
        self,
        module_under_test: str,
    ) -> Iterator[AsyncMock]:
        with patch(
            f"{module_under_test}.check_prompt_registration_status_after_it_finishes",
            new_callable=AsyncMock,
        ) as mock_func:
            yield mock_func

    @pytest.mark.asyncio
    async def test_check_prompt_registration_status_after_it_finishes_succeeds(
        self,
        mock_datarobot_mcp_server: Mock,
    ) -> None:
        mock_function_name = "sadfads"
        mock_registered_prompt = Mock()
        mock_registered_prompt.name = mock_function_name
        mock_datarobot_mcp_server.get_prompts.return_value = {"adsfdas": mock_registered_prompt}

        await check_prompt_registration_status_after_it_finishes(
            mock_datarobot_mcp_server,
            mock_function_name,
        )

        mock_datarobot_mcp_server.get_prompts.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_prompt_registration_status_after_it_finishes_fails(
        self,
        mock_datarobot_mcp_server: Mock,
    ) -> None:
        mock_datarobot_mcp_server.get_prompts.return_value = {"adsfdas": Mock()}

        with pytest.raises(RuntimeError):
            await check_prompt_registration_status_after_it_finishes(
                mock_datarobot_mcp_server,
                Mock(),
            )
            mock_datarobot_mcp_server.get_prompts.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_prompt_with_proper_tool_category(
        self,
        mock_check_prompt_registration_status_after_it_finishes: Mock,
        mock_datarobot_mcp_server: Mock,
        mock_dr_mcp_extras: Mock,
        mock_get_prompt_name_no_duplicate: AsyncMock,
        mock_mcp_tool_callable: Mock,
        mock_prompt_from_function: Mock,
    ) -> None:
        prompt_func_name = Mock()
        actual_output = await register_prompt(fn=mock_mcp_tool_callable, name=prompt_func_name)

        mock_dr_mcp_extras.assert_called_once_with(type="prompt")
        mock_dr_mcp_extras_decorator = mock_dr_mcp_extras.return_value
        mock_dr_mcp_extras_decorator.assert_called_once_with(mock_mcp_tool_callable)
        mock_get_prompt_name_no_duplicate.assert_called_once_with(
            mock_datarobot_mcp_server,
            prompt_func_name,
        )
        mock_wrapper_func = mock_dr_mcp_extras_decorator.return_value
        mock_prompt_from_function.assert_called_once_with(
            fn=mock_wrapper_func,
            name=mock_get_prompt_name_no_duplicate.return_value,
            title=None,
            description=None,
            tags=None,
            meta={"prompt_category": DataRobotMCPPromptCategory.USER_PROMPT_TEMPLATE_VERSION.name},
        )
        mock_datarobot_mcp_server.add_prompt.assert_called_once_with(
            mock_prompt_from_function.return_value
        )
        mock_check_prompt_registration_status_after_it_finishes.assert_called_once_with(
            mock_datarobot_mcp_server,
            mock_get_prompt_name_no_duplicate.return_value,
        )
        mock_datarobot_mcp_server.notify_prompts_changed.assert_called_once_with()
        assert actual_output == mock_datarobot_mcp_server.add_prompt.return_value

    @pytest.mark.asyncio
    async def test_register_prompt_sets_mapping_with_deduplicated_name(
        self,
        mock_check_prompt_registration_status_after_it_finishes: Mock,
        mock_datarobot_mcp_server: Mock,
        mock_dr_mcp_extras: Mock,
        mock_get_prompt_name_no_duplicate: AsyncMock,
        mock_mcp_tool_callable: Mock,
        mock_prompt_from_function: Mock,
    ) -> None:
        prompt_func_name = "my_prompt"
        deduplicated_name = "my_prompt (abcd)"
        mock_get_prompt_name_no_duplicate.return_value = deduplicated_name
        mock_datarobot_mcp_server.get_prompt_mapping = AsyncMock(return_value={})
        mock_datarobot_mcp_server.set_prompt_mapping = AsyncMock()

        await register_prompt(
            fn=mock_mcp_tool_callable,
            name=prompt_func_name,
            prompt_template=("pt1", "v1"),
        )

        mock_datarobot_mcp_server.set_prompt_mapping.assert_called_once_with(
            "pt1", "v1", deduplicated_name
        )
        mock_prompt_from_function.assert_called_once_with(
            fn=mock_dr_mcp_extras.return_value.return_value,
            name=deduplicated_name,
            title=None,
            description=None,
            tags=None,
            meta={"prompt_category": DataRobotMCPPromptCategory.USER_PROMPT_TEMPLATE_VERSION.name},
        )

    @pytest.mark.asyncio
    async def test_register_prompt_sets_mapping_when_prompt_template_provided(
        self,
        mock_check_prompt_registration_status_after_it_finishes: Mock,
        mock_datarobot_mcp_server: Mock,
        mock_dr_mcp_extras: Mock,
        mock_get_prompt_name_no_duplicate: AsyncMock,
        mock_mcp_tool_callable: Mock,
        mock_prompt_from_function: Mock,
    ) -> None:
        deduplicated_name = "my_prompt"
        mock_get_prompt_name_no_duplicate.return_value = deduplicated_name
        mock_datarobot_mcp_server.set_prompt_mapping = AsyncMock()

        await register_prompt(
            fn=mock_mcp_tool_callable,
            name="my_prompt",
            prompt_template=("pt1", "v1"),
        )

        mock_datarobot_mcp_server.set_prompt_mapping.assert_called_once_with(
            "pt1", "v1", deduplicated_name
        )
        mock_get_prompt_name_no_duplicate.assert_called_once_with(
            mock_datarobot_mcp_server, "my_prompt"
        )

    @pytest.mark.asyncio
    async def test_register_prompt_meta_is_readable_by_lineage(self) -> None:
        """
        GIVEN a prompt registered via register_prompt on a live server
        WHEN lineage builds MCPPromptMetadata from the registered FastMCP prompt
        THEN the category is read from meta without error
        (regression: register_prompt stamped the category under 'resource_category'
        while lineage reads 'prompt_category', so every lineage sync over a
        dynamically registered prompt crashed with KeyError).
        """
        test_mcp = DataRobotMCP()

        async def greet() -> str:
            return "hi"

        with patch("datarobot_genai.drmcp.core.mcp_instance.mcp", test_mcp):
            await register_prompt(fn=greet, name="greet")
            prompts = await test_mcp.get_prompts()

        registered = next(p for p in prompts.values() if p.name == "greet")
        metadata = MCPPromptMetadata.from_fastmcp_item(registered)

        assert metadata.name == "greet"
        assert metadata.type == DataRobotMCPPromptCategory.USER_PROMPT_TEMPLATE_VERSION.name
