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

from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import (
    get_datarobot_prompt_template_and_version,
)
from datarobot_genai.drmcp.core.dynamic_prompts.register import (
    register_prompt_from_datarobot_prompt_management,
)
from datarobot_genai.drmcp.core.exceptions import DynamicPromptRegistrationError
from datarobot_genai.drmcp.core.mcp_instance import mcp

logger = logging.getLogger(__name__)


async def register_prompt_for_prompt_template_id_and_version(
    prompt_template_id: str, prompt_template_version_id: str
) -> Prompt:
    """Register a Prompt for a specific prompt template ID and version.

    Args:
        prompt_template_id: The ID of the DataRobot prompt template.
        prompt_template_version_id: The ID of the DataRobot prompt template version.

    Raises
    ------
        DynamicPromptRegistrationError: If registration fails at any step.

    Returns
    -------
        The registered Prompt instance.
    """
    # Temporary and slower until SDK with prompt templates is not released
    prompt_with_version = get_datarobot_prompt_template_and_version(
        prompt_template_id, prompt_template_version_id
    )

    if not prompt_with_version:
        raise DynamicPromptRegistrationError

    prompt_template, prompt_template_version = prompt_with_version

    registered_prompt = await register_prompt_from_datarobot_prompt_management(
        prompt_template=prompt_template, prompt_template_version=prompt_template_version
    )
    return registered_prompt


async def get_registered_prompt_templates() -> dict[str, tuple[str, str]]:
    """Get all registered prompt templates in the MCP instance."""
    prompts = await mcp.get_prompt_mapping()
    return prompts


async def delete_registered_prompt_template(prompt_template_id: str) -> bool:
    """Delete the prompt registered for the prompt template id in the MCP instance."""
    prompt_templates = await mcp.get_prompt_mapping()
    if prompt_template_id not in prompt_templates:
        logger.debug(f"No prompt registered for prompt template id {prompt_template_id}")
        return False

    prompt_template_version_id, prompt_name = prompt_templates[prompt_template_id]
    await mcp.remove_prompt_mapping(prompt_template_id, prompt_template_version_id)
    logger.info(
        f"Deleted prompt name {prompt_name} for prompt template id {prompt_template_id}, "
        f"version {prompt_template_version_id}"
    )
    return True
