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

from dataclasses import dataclass

import datarobot as dr

from datarobot_genai.drmcp.core.clients import get_api_client

# Needed SDK version (3.10.0) is not published yet. We'll reimplement simplified version of it.
# get_datarobot_prompt_templates = dr.genai.PromptTemplate.list()
# DrPrompt = dr.genai.PromptTemplate
# DrPromptVersion = dr.genai.PromptTemplateVersion
# DrVariable = dr.genai.Variable


@dataclass
class DrVariable:
    name: str
    description: str


@dataclass
class DrPromptVersion:
    id: str
    version: int
    prompt_text: str
    variables: list[DrVariable]


@dataclass
class DrPrompt:
    id: str
    name: str
    description: str

    def get_latest_version(self) -> DrPromptVersion | None:
        prompt_template_versions = get_datarobot_prompt_template_versions(self.id)
        if not prompt_template_versions:
            return None
        latest_version = max(prompt_template_versions, key=lambda v: v.version)
        return latest_version


def get_datarobot_prompt_templates() -> list[DrPrompt]:
    prompt_templates_data = dr.utils.pagination.unpaginate(
        initial_url="genai/promptTemplates/", initial_params={}, client=get_api_client()
    )

    return [
        DrPrompt(
            id=prompt_template["id"],
            name=prompt_template["name"],
            description=prompt_template["description"],
        )
        for prompt_template in prompt_templates_data
    ]


def get_datarobot_prompt_template_versions(prompt_template_id: str) -> list[DrPromptVersion]:
    prompt_template_versions_data = dr.utils.pagination.unpaginate(
        initial_url=f"genai/promptTemplates/{prompt_template_id}/versions/",
        initial_params={},
        client=get_api_client(),
    )
    prompt_template_versions = []
    for prompt_template_version in prompt_template_versions_data:
        variables = [
            DrVariable(name=v["name"], description=v["description"])
            for v in prompt_template_version["variables"]
        ]
        prompt_template_versions.append(
            DrPromptVersion(
                id=prompt_template_version["id"],
                version=prompt_template_version["version"],
                prompt_text=prompt_template_version["promptText"],
                variables=variables,
            )
        )
    return prompt_template_versions
