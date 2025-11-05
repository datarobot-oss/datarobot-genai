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
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import DrPrompt
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import DrPromptVersion
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import DrVariable
from datarobot_genai.drmcp.core.dynamic_prompts.register import (
    register_prompts_from_datarobot_prompt_management,
)
from datarobot_genai.drmcp.core.mcp_instance import mcp


@pytest.fixture
def prompt_template_id_ok() -> str:
    return "69086ea4834952718366b2ce"


@pytest.fixture
def prompt_template_version_id_ok() -> str:
    return "69086ea4b65d70489c5b198d"


@pytest.fixture
def get_prompt_template_mock(prompt_template_id_ok: str, prompt_template_version_id_ok: str):
    """Set up all API endpoint mocks."""
    dr_prompt_version = DrPromptVersion(
        id=prompt_template_version_id_ok,
        version=3,
        prompt_text="Write greeting for {{name}} in max {{sentences}} sentences.",
        variables=[
            DrVariable(name="name", description="Person name"),
            DrVariable(name="sentences", description="Number of sentences"),
        ],
    )
    dr_prompt = DrPrompt(
        id=prompt_template_id_ok,
        name="Dummy prompt name",
        description="Dummy description",
    )
    dr_prompt.get_latest_version = lambda: dr_prompt_version

    with patch(
        "datarobot_genai.drmcp.core.dynamic_prompts.register.get_datarobot_prompt_templates",
        return_value=[dr_prompt],
    ):
        yield


@pytest.mark.asyncio
async def test_register_prompt_from_datarobot_prompt_management(
    get_prompt_template_mock,
) -> None:
    await register_prompts_from_datarobot_prompt_management()

    prompts = {prompt for prompt in await mcp.get_prompts()}

    assert "Dummy prompt name" in prompts, "`Dummy prompt name` is missing."
