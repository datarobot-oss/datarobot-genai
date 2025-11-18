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

from datarobot_genai.drmcp.core.mcp_instance import mcp

logger = logging.getLogger(__name__)


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
