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

import pytest

from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import DrPrompt
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import DrPromptVersion
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import DrVariable
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import get_datarobot_prompt_template
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import get_datarobot_prompt_template_version
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import get_datarobot_prompt_template_versions
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import get_datarobot_prompt_templates


class TestDrLib:
    """Tests for dr_lib - happy path."""

    @pytest.mark.asyncio
    async def test_get_prompt_templates(self) -> None:
        """Test get prompt templates."""
        with (
            patch(
                "datarobot_genai.drmcp.core.dynamic_prompts.dr_lib.dr",
            ) as mock_dr,
            patch("datarobot_genai.drmcp.core.dynamic_prompts.dr_lib.get_api_client"),
        ):
            mock_dr.utils.pagination.unpaginate = Mock(
                return_value=[
                    {
                        "id": "69086ea4b65d70489c5b198d",
                        "name": "Prompt template 1",
                        "description": "Desc 1",
                    },
                    {
                        "id": "79086ea4b65d70489c5b198d",
                        "name": "Prompt template 2",
                        "description": "Desc 2",
                    },
                ]
            )

            prompt_templates = get_datarobot_prompt_templates()

        assert len(prompt_templates) == 2
        assert prompt_templates == [
            DrPrompt(id="69086ea4b65d70489c5b198d", name="Prompt template 1", description="Desc 1"),
            DrPrompt(id="79086ea4b65d70489c5b198d", name="Prompt template 2", description="Desc 2"),
        ]

    @pytest.mark.asyncio
    async def test_get_prompt_templates_versions(self) -> None:
        """Test get prompt templates versions."""
        prompt_template_id = "69086ea4b65d70489c5b198d"
        with (
            patch(
                "datarobot_genai.drmcp.core.dynamic_prompts.dr_lib.dr",
            ) as mock_dr,
            patch("datarobot_genai.drmcp.core.dynamic_prompts.dr_lib.get_api_client"),
        ):
            mock_dr.utils.pagination.unpaginate = Mock(
                return_value=[
                    {
                        "id": "89086ea4b65d70489c5b198d",
                        "version": 2,
                        "promptText": "Text with {{variable}}.",
                        "variables": [{"name": "variable", "description": "Dummy variable"}],
                    },
                ]
            )

            prompt_template_versions = get_datarobot_prompt_template_versions(prompt_template_id)

        assert len(prompt_template_versions) == 1
        assert prompt_template_versions == [
            DrPromptVersion(
                id="89086ea4b65d70489c5b198d",
                version=2,
                prompt_text="Text with {{variable}}.",
                variables=[DrVariable(name="variable", description="Dummy variable")],
            ),
        ]

    @pytest.mark.asyncio
    async def test_get_latest_prompt_template_version(self) -> None:
        """Test get latest prompt template version -- happy path."""
        prompt_template = DrPrompt(
            id="69086ea4b65d70489c5b198d", name="Prompt template 1", description="Desc 1"
        )

        with (
            patch(
                "datarobot_genai.drmcp.core.dynamic_prompts.dr_lib.dr",
            ) as mock_dr,
            patch("datarobot_genai.drmcp.core.dynamic_prompts.dr_lib.get_api_client"),
        ):
            mock_dr.utils.pagination.unpaginate = Mock(
                return_value=[
                    {
                        "id": "89086ea4b65d70489c5b198d",
                        "version": 2,
                        "promptText": "Text with {{variable}}.",
                        "variables": [{"name": "variable", "description": "Dummy variable"}],
                    },
                ]
            )

            latest_prompt_template_version = prompt_template.get_latest_version()

        assert latest_prompt_template_version == DrPromptVersion(
            id="89086ea4b65d70489c5b198d",
            version=2,
            prompt_text="Text with {{variable}}.",
            variables=[DrVariable(name="variable", description="Dummy variable")],
        )

    @pytest.mark.asyncio
    async def test_get_latest_prompt_template_version_when_no_versions(self) -> None:
        """Test get latest prompt template version when no versions."""
        prompt_template = DrPrompt(
            id="69086ea4b65d70489c5b198d", name="Prompt template 1", description="Desc 1"
        )

        with (
            patch(
                "datarobot_genai.drmcp.core.dynamic_prompts.dr_lib.dr",
            ) as mock_dr,
            patch("datarobot_genai.drmcp.core.dynamic_prompts.dr_lib.get_api_client"),
        ):
            mock_dr.utils.pagination.unpaginate = Mock(return_value=[])

            latest_prompt_template_version = prompt_template.get_latest_version()

        assert latest_prompt_template_version is None

    @pytest.mark.asyncio
    async def test_get_prompt_template(self) -> None:
        """Test get prompt template."""
        prompt_template_get_mock = Mock()
        prompt_template_get_mock.json.return_value = {
            "id": "69086ea4b65d70489c5b198d",
            "name": "Prompt template 1",
            "description": "Desc 1",
        }

        with (
            patch(
                "datarobot_genai.drmcp.core.dynamic_prompts.dr_lib.get_api_client"
            ) as mock_api_client,
        ):
            mock_api_client.return_value.get.return_value = prompt_template_get_mock

            prompt_template = get_datarobot_prompt_template(
                prompt_template_id="69086ea4b65d70489c5b198d",
            )

        assert prompt_template == DrPrompt(
            id="69086ea4b65d70489c5b198d", name="Prompt template 1", description="Desc 1"
        )

    @pytest.mark.asyncio
    async def test_get_prompt_template_version(self) -> None:
        """Test get prompt template version."""
        prompt_template_version_get_mock = Mock()
        prompt_template_version_get_mock.json.return_value = {
            "id": "89086ea4b65d70489c5b198d",
            "version": 2,
            "promptText": "Text with {{variable}}.",
            "variables": [{"name": "variable", "description": "Dummy variable"}],
        }

        with (
            patch(
                "datarobot_genai.drmcp.core.dynamic_prompts.dr_lib.get_api_client"
            ) as mock_api_client,
        ):
            mock_api_client.return_value.get.return_value = prompt_template_version_get_mock

            prompt_template_version = get_datarobot_prompt_template_version(
                prompt_template_id="69086ea4b65d70489c5b198d",
                prompt_template_version_id="89086ea4b65d70489c5b198d",
            )

        assert prompt_template_version == DrPromptVersion(
            id="89086ea4b65d70489c5b198d",
            version=2,
            prompt_text="Text with {{variable}}.",
            variables=[DrVariable(name="variable", description="Dummy variable")],
        )
