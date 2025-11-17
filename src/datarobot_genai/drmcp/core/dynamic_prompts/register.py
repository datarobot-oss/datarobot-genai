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
import keyword
import logging
import re
from collections.abc import Callable
from inspect import Parameter
from inspect import Signature

from fastmcp.prompts.prompt import Prompt
from pydantic import Field

from datarobot_genai.drmcp.core.exceptions import DynamicPromptRegistrationError
from datarobot_genai.drmcp.core.mcp_instance import register_prompt

from .dr_lib import DrPrompt
from .dr_lib import DrVariable
from .dr_lib import get_datarobot_prompt_templates

logger = logging.getLogger(__name__)


async def register_prompts_from_datarobot_prompt_management() -> None:
    """Register prompts from DataRobot Prompt Management."""
    prompts = get_datarobot_prompt_templates()
    logger.info(f"Found {len(prompts)} prompts in Prompts Management.")

    # Try to register each prompt, continue on failure
    for prompt in prompts:
        try:
            await register_prompt_from_datarobot_prompt_management(prompt)
        except DynamicPromptRegistrationError:
            pass


async def register_prompt_from_datarobot_prompt_management(
    prompt: DrPrompt,
) -> Prompt:
    """Register a single prompt.

    Args:
        prompt: The prompt within DataRobot Prompt Management.

    Raises
    ------
        DynamicPromptRegistrationError: If registration fails at any step.

    Returns
    -------
        The registered Prompt instance.
    """
    latest_version = prompt.get_latest_version()

    if latest_version is None:
        logger.info(f"No latest version in Prompts Management for prompt id: {prompt.id}")
        raise DynamicPromptRegistrationError

    logger.info(
        f"Found prompt: id: {prompt.id}, "
        f"name: {prompt.name}, "
        f"latest version id: {latest_version.id}, "
        f"version: {latest_version.version}."
    )

    try:
        valid_fn_name = to_valid_mcp_prompt_name(prompt.name)
    except ValueError as e:
        raise DynamicPromptRegistrationError from e

    prompt_fn = make_prompt_function(
        name=valid_fn_name,
        description=prompt.description,
        prompt_text=latest_version.prompt_text,
        variables=latest_version.variables,
    )

    try:
        # Register using generic external tool registration with the config
        return await register_prompt(
            fn=prompt_fn,
            name=prompt.name,
            description=prompt.description,
            meta={"prompt_template_id": prompt.id, "prompt_template_version_id": latest_version.id},
        )

    except Exception as exc:
        logger.error(f"Skipping prompt {prompt.id}. Registration failed: {exc}")
        raise DynamicPromptRegistrationError(
            "Registration failed. Could not create prompt."
        ) from exc


def to_valid_mcp_prompt_name(s: str) -> str:
    """Convert an arbitrary string into a valid MCP prompt name."""
    # Replace any sequence of invalid characters with '_'
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)

    # Remove leading characters that are not letters or underscores (can't start with a digit or _)
    s = re.sub(r"^[^a-zA-Z]+", "", s)

    # Remove following _
    s = re.sub(r"_+$", "", s)

    # If string is empty after cleaning, raise error
    if not s:
        raise ValueError(f"Cannot convert {s} to valid MCP prompt name.")

    # Make sure it’s a valid identifier and not a reserved keyword
    if keyword.iskeyword(s) or not s.isidentifier():
        s = f"{s}_prompt"

    return s


def make_prompt_function(
    name: str, description: str, prompt_text: str, variables: list[DrVariable]
) -> Callable:
    params = [
        Parameter(
            name=v.name,
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            default=Field(description=v.description),
        )
        for v in variables
    ]

    async def template_function(**kwargs) -> str:  # type: ignore
        prompt_text_correct = prompt_text.replace("{{", "{").replace("}}", "}")
        try:
            return prompt_text_correct.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable {e.args[0]} for prompt '{name}'")

    # Apply metadata
    template_function.__name__ = name
    template_function.__doc__ = description
    template_function.__signature__ = Signature(params)  # type: ignore

    return template_function
