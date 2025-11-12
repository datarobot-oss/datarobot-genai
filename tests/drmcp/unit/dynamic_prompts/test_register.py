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

import pytest

from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import DrVariable
from datarobot_genai.drmcp.core.dynamic_prompts.register import make_prompt_function
from datarobot_genai.drmcp.core.dynamic_prompts.register import (
    register_prompts_from_datarobot_prompt_management,
)
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
            DrVariable(name="variable_a", description="variable_a_desc"),
            DrVariable(name="variable_b", description="variable_b_desc"),
        ]

        prompt_function = make_prompt_function(name, description, prompt_text, variables)
        prompt = await prompt_function(variable_a="variable_a_value", variable_b="variable_b_value")

        assert prompt == "dummy prompt text variable_a_value and variable_b_value"


class TestRegisterPrompt:
    """Tests for register_prompt - happy path."""

    @pytest.mark.asyncio
    async def test_register_prompt_from_datarobot_prompt_management(
        self,
        get_prompt_template_mock: None,
    ) -> None:
        """Test register prompt from dr prompt mgmt."""
        await register_prompts_from_datarobot_prompt_management()

        prompts = {prompt for prompt in await mcp.get_prompts()}

        assert "Dummy prompt name" in prompts, "`Dummy prompt name` is missing."
