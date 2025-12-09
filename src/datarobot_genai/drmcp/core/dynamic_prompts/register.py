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

import datarobot as dr
from fastmcp.prompts.prompt import Prompt
from pydantic import Field

from datarobot_genai.drmcp.core.exceptions import DynamicPromptRegistrationError
from datarobot_genai.drmcp.core.mcp_instance import register_prompt

from .dr_lib import get_datarobot_prompt_template_versions
from .dr_lib import get_datarobot_prompt_templates

logger = logging.getLogger(__name__)


async def register_prompts_from_datarobot_prompt_management() -> None:
    """Register prompts from DataRobot Prompt Management."""
    prompts = get_datarobot_prompt_templates()
    logger.info(f"Found {len(prompts)} prompts in Prompts Management.")
    all_prompts_versions = get_datarobot_prompt_template_versions(
        prompt_template_ids=list({prompt.id for prompt in prompts})
    )

    # Try to register each prompt, continue on failure
    for prompt in prompts:
        prompt_versions = all_prompts_versions.get(prompt.id)
        if not prompt_versions:
            logger.warning(f"Prompt template id {prompt.id} has no versions.")
            continue

        latest_version = max(prompt_versions, key=lambda v: v.version)

        try:
            await register_prompt_from_datarobot_prompt_management(prompt, latest_version)
        except DynamicPromptRegistrationError:
            pass


async def register_prompt_from_datarobot_prompt_management(
    prompt_template: dr.genai.PromptTemplate,
    prompt_template_version: dr.genai.PromptTemplateVersion | None = None,
) -> Prompt:
    """Register a single prompt.

    Args:
        prompt_template: The prompt within DataRobot Prompt Management.
        prompt_template_version: Optional prompt version within DataRobot Prompt Management.
            If not provided -- latest version will be used

    Raises
    ------
        DynamicPromptRegistrationError: If registration fails at any step.

    Returns
    -------
        The registered Prompt instance.
    """
    if not prompt_template_version:
        prompt_template_version_to_register = prompt_template.get_latest_version()

        if prompt_template_version_to_register is None:
            logger.info(
                f"No latest version in Prompts Management for prompt id: {prompt_template.id}"
            )
            raise DynamicPromptRegistrationError

    else:
        prompt_template_version_to_register = prompt_template_version

    logger.info(
        f"Found prompt: id: {prompt_template.id}, "
        f"name: {prompt_template.name}, "
        f"prompt version id: {prompt_template_version_to_register.id}, "
        f"version: {prompt_template_version_to_register.version}."
    )

    try:
        valid_fn_name = to_valid_mcp_prompt_name(prompt_template.name)
    except ValueError as e:
        raise DynamicPromptRegistrationError from e

    prompt_fn = make_prompt_function(
        name=valid_fn_name,
        description=prompt_template.description,
        prompt_text=prompt_template_version_to_register.prompt_text,
        variables=prompt_template_version_to_register.variables,
    )

    try:
        # Register using generic external tool registration with the config
        return await register_prompt(
            fn=prompt_fn,
            name=prompt_template.name,
            description=prompt_template.description,
            meta={
                "prompt_template_id": prompt_template.id,
                "prompt_template_version_id": prompt_template_version_to_register.id,
            },
            prompt_template=(prompt_template.id, prompt_template_version_to_register.id),
        )

    except Exception as exc:
        logger.error(f"Skipping prompt {prompt_template.id}. Registration failed: {exc}")
        raise DynamicPromptRegistrationError(
            "Registration failed. Could not create prompt."
        ) from exc


def _escape_non_ascii(s: str) -> str:
    out = []
    for ch in s:
        # If its space -> change to underscore
        if ch.isspace():
            out.append("_")
        # ASCII letter, digit or underscore -> keep
        elif ch.isascii() and (ch.isalnum() or ch == "_"):
            out.append(ch)
        # Everything else -> encode as 'xHEX'
        else:
            out.append(f"x{ord(ch):x}")
    return "".join(out)


def to_valid_mcp_prompt_name(s: str) -> str:
    """Convert an arbitrary string into a valid MCP prompt name."""
    # If its ONLY numbers return "prompt_[number]"
    if s.isdigit():
        return f"prompt_{s}"

    # First, ASCII-transliterate using hex escape for non-ASCII
    if not s.isascii():
        # whole string non-ascii? -> escape and prefix with prompt_
        encoded = _escape_non_ascii(s)
        return f"prompt_{encoded}"

    # Replace any sequence of invalid characters with '_'
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)

    # Remove leading characters that are not letters or underscores (can't start with a digit or _)
    s = re.sub(r"^[^a-zA-Z]+", "", s)

    # Remove following _
    s = re.sub(r"_+$", "", s)

    # If string is empty after cleaning, raise error
    if not s:
        raise ValueError(f"Cannot convert {s} to valid MCP prompt name.")

    # Make sure it's a valid identifier and not a reserved keyword
    if keyword.iskeyword(s) or not s.isidentifier():
        s = f"{s}_prompt"

    return s


def make_prompt_function(
    name: str, description: str, prompt_text: str, variables: list[dr.genai.Variable]
) -> Callable:
    params = []
    for v in variables:
        if keyword.iskeyword(v.name):
            raise ValueError(f"Variable name '{v.name}' is invalid.")

        try:
            param = Parameter(
                name=v.name,
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Field(description=v.description),
            )
        except ValueError as e:
            raise ValueError(f"Variable name '{v.name}' is invalid.") from e

        params.append(param)

    async def template_function(**kwargs) -> str:  # type: ignore
        prompt_text_correct = prompt_text.replace("{{", "{").replace("}}", "}")
        try:
            return prompt_text_correct.format(**kwargs)
        except KeyError as exc:
            raise ValueError(f"Missing variable {exc.args[0]} for prompt '{name}'") from exc

    # Apply metadata
    template_function.__name__ = name
    template_function.__doc__ = description
    template_function.__signature__ = Signature(params)  # type: ignore

    return template_function
