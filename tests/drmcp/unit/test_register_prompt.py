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
            meta={
                "resource_category": DataRobotMCPPromptCategory.USER_PROMPT_TEMPLATE_VERSION.name
            },
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
