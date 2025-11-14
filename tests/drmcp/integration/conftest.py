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
from typing import Any

import pytest

from datarobot_genai.drmcp.test_utils.prompt_management import get_or_create_prompt_template
from datarobot_genai.drmcp.test_utils.prompt_management import get_or_create_prompt_template_version


@pytest.fixture(scope="session")
def prompt_template_name_without_version() -> str:
    return "drmcp-integration-test-prompt-without-version"


@pytest.fixture(scope="session")
def prompt_template_name_with_version_without_variables() -> str:
    return "drmcp-integration-test-prompt-with-version-without-variables"


@pytest.fixture(scope="session")
def prompt_template_text_without_variables() -> str:
    return "Prompt text without any variables."


@pytest.fixture(scope="session")
def prompt_template_name_with_version_with_variables() -> str:
    return "drmcp-integration-test-prompt-with-variables"


@pytest.fixture(scope="session")
def prompt_template_text_with_2_variables() -> str:
    return "Prompt text to greet {{name}} in max {{sentences}} sentences."


@pytest.fixture(scope="session")
def prompt_template_without_versions(prompt_template_name_without_version: str) -> dict[str, Any]:
    return get_or_create_prompt_template(prompt_template_name_without_version)


@pytest.fixture(scope="session")
def prompt_template_with_version_without_variables(
    prompt_template_name_with_version_without_variables: str,
    prompt_template_text_without_variables: str,
) -> dict[str, Any]:
    prompt_template = get_or_create_prompt_template(
        prompt_template_name_with_version_without_variables
    )
    prompt_template_version = get_or_create_prompt_template_version(
        prompt_template_id=prompt_template["id"],
        prompt_text=prompt_template_text_without_variables,
        variables=[],
    )
    return {
        "id": prompt_template["id"],
        "name": prompt_template_name_with_version_without_variables,
        "version_id": prompt_template_version["id"],
        "prompt_text": prompt_template_version["prompt_text"],
    }


@pytest.fixture(scope="session")
def prompt_template_with_version_with_variables(
    prompt_template_name_with_version_with_variables: str,
    prompt_template_text_with_2_variables: str,
) -> dict[str, Any]:
    prompt_template = get_or_create_prompt_template(
        prompt_template_name_with_version_with_variables
    )
    prompt_template_version = get_or_create_prompt_template_version(
        prompt_template_id=prompt_template["id"],
        prompt_text=prompt_template_text_with_2_variables,
        variables=["name", "sentences"],
    )
    return {
        "id": prompt_template["id"],
        "name": prompt_template_name_with_version_with_variables,
        "version_id": prompt_template_version["id"],
        "prompt_text": prompt_template_version["prompt_text"],
    }
