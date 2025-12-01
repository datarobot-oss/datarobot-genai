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

import logging

from fastmcp.prompts.prompt import Prompt

from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import get_datarobot_prompt_template
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import get_datarobot_prompt_template_version
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import get_datarobot_prompt_template_versions
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import get_datarobot_prompt_templates
from datarobot_genai.drmcp.core.dynamic_prompts.register import (
    register_prompt_from_datarobot_prompt_management,
)
from datarobot_genai.drmcp.core.exceptions import DynamicPromptRegistrationError
from datarobot_genai.drmcp.core.mcp_instance import mcp

logger = logging.getLogger(__name__)


async def register_prompt_from_prompt_template_id_and_version(
    prompt_template_id: str, prompt_template_version_id: str | None
) -> Prompt:
    """Register a Prompt for a specific prompt template ID and version.

    Args:
        prompt_template_id: The ID of the DataRobot prompt template.
        prompt_template_version_id: Optional ID of the DataRobot prompt template version.
            If not provided latest will be used

    Raises
    ------
        DynamicPromptRegistrationError: If registration fails at any step.

    Returns
    -------
        The registered Prompt instance.
    """
    prompt_template = get_datarobot_prompt_template(prompt_template_id)

    if not prompt_template:
        raise DynamicPromptRegistrationError("Registration failed. Could not find prompt template.")

    if not prompt_template_version_id:
        return await register_prompt_from_datarobot_prompt_management(
            prompt_template=prompt_template
        )

    prompt_template_version = get_datarobot_prompt_template_version(
        prompt_template_id, prompt_template_version_id
    )

    if not prompt_template_version:
        raise DynamicPromptRegistrationError(
            "Registration failed. Could not find prompt template version."
        )

    return await register_prompt_from_datarobot_prompt_management(
        prompt_template=prompt_template, prompt_template_version=prompt_template_version
    )


async def delete_registered_prompt_template(prompt_template_id: str) -> bool:
    """Delete the prompt registered for the prompt template id in the MCP instance."""
    prompt_templates_mappings = await mcp.get_prompt_mapping()
    if prompt_template_id not in prompt_templates_mappings:
        logger.debug(f"No prompt registered for prompt template id {prompt_template_id}")
        return False

    prompt_template_version_id, prompt_name = prompt_templates_mappings[prompt_template_id]
    await mcp.remove_prompt_mapping(prompt_template_id, prompt_template_version_id)
    logger.info(
        f"Deleted prompt name {prompt_name} for prompt template id {prompt_template_id}, "
        f"version {prompt_template_version_id}"
    )
    return True


async def refresh_registered_prompt_template() -> None:
    """Refresh all registered prompt templates in the MCP instance."""
    prompt_templates = get_datarobot_prompt_templates()
    prompt_templates_ids = {p.id for p in prompt_templates}
    prompt_templates_versions = get_datarobot_prompt_template_versions(list(prompt_templates_ids))

    mcp_prompt_templates_mappings = await mcp.get_prompt_mapping()

    for prompt_template in prompt_templates:
        prompt_template_versions = prompt_templates_versions.get(prompt_template.id)
        if not prompt_template_versions:
            continue

        latest_version = max(prompt_template_versions, key=lambda v: v.version)

        if prompt_template.id not in mcp_prompt_templates_mappings:
            # New prompt template -> add
            await register_prompt_from_datarobot_prompt_management(
                prompt_template=prompt_template, prompt_template_version=latest_version
            )
            continue

        mcp_prompt_template_version, mcp_prompt = mcp_prompt_templates_mappings[prompt_template.id]

        if mcp_prompt_template_version != latest_version:
            # Current version saved in MCP is not the latest one => update it
            await register_prompt_from_datarobot_prompt_management(
                prompt_template=prompt_template, prompt_template_version=latest_version
            )
            continue

        # Else => mcp_prompt_template_version == latest_version
        # For now it means nothing changed as there's no possibility to edit promp template version.

    for mcp_prompt_template_id, (
        mcp_prompt_template_version_id,
        _,
    ) in mcp_prompt_templates_mappings.items():
        if mcp_prompt_template_id not in prompt_templates_ids:
            # We need to also delete prompt templates that are
            await mcp.remove_prompt_mapping(mcp_prompt_template_id, mcp_prompt_template_version_id)
