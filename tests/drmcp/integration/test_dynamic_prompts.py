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
import re

import pytest
from mcp import McpError

from datarobot_genai.drmcp import integration_test_mcp_session
from datarobot_genai.drmcp.core.mcp_instance import DataRobotMCP


@pytest.mark.asyncio
class TestMCPDRPromptManagementIntegration:
    """Integration tests for MCP DR Prompt Management integration."""

    # Staging is slow sometimes, so higher timeout not to fail randomly
    TIMEOUT: int = 60

    async def test_prompt_without_version(self, prompt_template_without_versions: dict) -> None:
        """Integration test for prompt template without any versions that cannot be used in MCP."""
        async with integration_test_mcp_session(timeout=self.TIMEOUT, use_stub=False) as session:
            prompt_template_name = prompt_template_without_versions["name"]

            # Check if testing prompt
            # IS NOT loaded as it's "broken" one
            prompts_list = await session.list_prompts()
            assert prompt_template_name not in {p.name for p in prompts_list.prompts}

            # Prompt template without prompt template versions assigned cannot be used in MCP
            with pytest.raises(McpError) as e:
                _ = await session.get_prompt(name=prompt_template_name, arguments={})

            # Check if it tells that prompt does not exist
            assert (
                e.value.error.message
                == "Unknown prompt: drmcp-integration-test-prompt-without-version"
            )

    async def test_prompt_with_version_without_variables(
        self, prompt_template_with_version_without_variables: dict
    ) -> None:
        """Integration test for prompt template with version without any variables."""
        async with integration_test_mcp_session(timeout=self.TIMEOUT, use_stub=False) as session:
            prompt_template_name = prompt_template_with_version_without_variables["name"]
            prompt_template_prompt_text = prompt_template_with_version_without_variables[
                "prompt_text"
            ]

            # Check if testing prompt is in list of all prompts
            prompts_list = await session.list_prompts()
            assert prompt_template_name in {p.name for p in prompts_list.prompts}

            # Simple prompt template without any variables
            result = await session.get_prompt(name=prompt_template_name, arguments={})

            # Check if it's correctly formatted prompt
            assert len(result.messages) == 1
            assert result.messages[0].role == "user"
            assert result.messages[0].content.text == prompt_template_prompt_text

    async def test_prompt_with_version_with_variables_happy_path(
        self, prompt_template_with_version_with_variables: dict
    ) -> None:
        """Integration test for prompt template with version with variables -- happy path."""
        var_1_name = "name"
        var_1_value = "Tester"
        var_2_name = "sentences"
        var_2_value = "5"

        async with integration_test_mcp_session(timeout=self.TIMEOUT, use_stub=False) as session:
            prompt_template_name = prompt_template_with_version_with_variables["name"]
            prompt_template_prompt_text = prompt_template_with_version_with_variables["prompt_text"]

            # Check if testing prompt is in list of all prompts
            prompts_list = await session.list_prompts()
            assert prompt_template_name in {p.name for p in prompts_list.prompts}

            # Simple prompt template with 2 variables, and substituting 2 values
            result = await session.get_prompt(
                name=prompt_template_name,
                arguments={var_1_name: var_1_value, var_2_name: var_2_value},
            )

            # Check if it's correctly formatted prompt
            prompt_text_with_values = prompt_template_prompt_text.replace("{{name}}", var_1_value)
            prompt_text_with_values = prompt_text_with_values.replace("{{sentences}}", var_2_value)

            assert len(result.messages) == 1
            assert result.messages[0].role == "user"
            assert result.messages[0].content.text == prompt_text_with_values

    async def test_prompt_with_version_with_variables_when_not_enough_variables_provided(
        self, prompt_template_with_version_with_variables: dict
    ) -> None:
        """Integration test for prompt template with version with variables
        when not enough variables provided.
        """
        async with integration_test_mcp_session(timeout=self.TIMEOUT, use_stub=False) as session:
            prompt_template_name = prompt_template_with_version_with_variables["name"]

            # Check if testing prompt is in list of all prompts
            prompts_list = await session.list_prompts()
            assert prompt_template_name in {p.name for p in prompts_list.prompts}

            # Simple prompt template with 2 variables, and substituting nothing
            with pytest.raises(McpError) as e:
                _ = await session.get_prompt(name=prompt_template_name, arguments={})

            # Check if error suggests missing values
            error_msg_base = (
                "Error rendering prompt 'drmcp-integration-test-prompt-with-variables': "
            )
            try:
                assert (
                    error_msg_base + "Missing required arguments: {'sentences', 'name'}"
                    == e.value.error.message
                )
            except AssertionError:  # Sometimes order of arguments is different
                assert (
                    error_msg_base + "Missing required arguments: {'name', 'sentences'}"
                    == e.value.error.message
                )

    async def test_prompt_when_duplicated_names_exists(
        self, prompt_templates_with_duplicates: tuple[dict, dict]
    ) -> None:
        """Integration test for prompt template when duplicated names exist."""
        first_prompt_template, second_prompt_template = prompt_templates_with_duplicates
        async with integration_test_mcp_session(timeout=self.TIMEOUT, use_stub=False) as session:
            prompt_template_name_1 = first_prompt_template["name"]
            prompt_template_name_2 = second_prompt_template["name"]

            # Its name from API -> it's duplicated
            assert prompt_template_name_1 == prompt_template_name_2

            # Check if both prompts are in list of all prompts
            prompts_list = await session.list_prompts()
            prompts_names = {p.name for p in prompts_list.prompts}

            # One prompt is without any "suffix"
            assert prompt_template_name_1 in prompts_names

            # Second prompt has suffix
            # We cannot mock uuid here as it's separate process running MCP server
            pattern = re.compile(rf"{prompt_template_name_1} \([A-Za-z0-9]{{4}}\)")
            matches = [s for s in prompts_names if pattern.search(s)]

            assert len(matches) == 1


@pytest.mark.asyncio
async def test_mcp_prompts_mapping_methods():
    mcp = DataRobotMCP()

    await mcp.set_prompt_mapping("id_1", "v_id_1", "prompt_1")
    await mcp.set_prompt_mapping("id_2", "v_id_2", "prompt_2")

    prompts = await mcp.get_prompt_mapping()
    assert prompts == {"id_1": ("v_id_1", "prompt_1"), "id_2": ("v_id_2", "prompt_2")}

    # override existing mapping, the same version id
    await mcp.set_prompt_mapping("id_1", "v_id_1", "prompt_3")
    prompts = await mcp.get_prompt_mapping()
    assert prompts == {
        "id_1": ("v_id_1", "prompt_3"),  # Difference
        "id_2": ("v_id_2", "prompt_2"),
    }

    # override existing mapping, different version id
    await mcp.set_prompt_mapping("id_2", "v_id_4", "prompt_4")
    prompts = await mcp.get_prompt_mapping()
    assert prompts == {
        "id_1": ("v_id_1", "prompt_3"),
        "id_2": ("v_id_4", "prompt_4"),  # Difference
    }

    # delete not existing mapping
    await mcp.remove_prompt_mapping("id_99", "v_id_99")
    prompts = await mcp.get_prompt_mapping()
    assert prompts == {"id_1": ("v_id_1", "prompt_3"), "id_2": ("v_id_4", "prompt_4")}

    # delete mapping with given prompt id but not matching version id
    await mcp.remove_prompt_mapping("id_1", "v_id_99")
    prompts = await mcp.get_prompt_mapping()
    assert prompts == {"id_1": ("v_id_1", "prompt_3"), "id_2": ("v_id_4", "prompt_4")}

    # delete first mapping -- happy path
    await mcp.remove_prompt_mapping("id_1", "v_id_1")
    prompts = await mcp.get_prompt_mapping()
    assert prompts == {"id_2": ("v_id_4", "prompt_4")}

    # delete second mapping -- happy path
    await mcp.remove_prompt_mapping("id_2", "v_id_4")
    prompts = await mcp.get_prompt_mapping()
    assert prompts == {}
