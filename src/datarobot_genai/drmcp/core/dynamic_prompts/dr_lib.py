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

    @classmethod
    def from_dict(cls, d: dict) -> "DrPromptVersion":
        variables = [
            DrVariable(name=v["name"], description=v["description"]) for v in d["variables"]
        ]
        return cls(
            id=d["id"],
            version=d["version"],
            prompt_text=d["promptText"],
            variables=variables,
        )


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

    @classmethod
    def from_dict(cls, d: dict) -> "DrPrompt":
        return cls(id=d["id"], name=d["name"], description=d["description"])


def get_datarobot_prompt_templates() -> list[DrPrompt]:
    prompt_templates_data = dr.utils.pagination.unpaginate(
        initial_url="genai/promptTemplates/", initial_params={}, client=get_api_client()
    )

    return [DrPrompt.from_dict(prompt_template) for prompt_template in prompt_templates_data]


def get_datarobot_prompt_template_versions(prompt_template_id: str) -> list[DrPromptVersion]:
    prompt_template_versions_data = dr.utils.pagination.unpaginate(
        initial_url=f"genai/promptTemplates/{prompt_template_id}/versions/",
        initial_params={},
        client=get_api_client(),
    )
    prompt_template_versions = []
    for prompt_template_version in prompt_template_versions_data:
        prompt_template_versions.append(DrPromptVersion.from_dict(prompt_template_version))
    return prompt_template_versions


def get_datarobot_prompt_template(prompt_template_id: str) -> DrPrompt | None:
    api_client = get_api_client()
    try:
        prompt_template_response = api_client.get(
            f"genai/promptTemplates/{prompt_template_id}/", join_endpoint=True
        )
        prompt_template_json = prompt_template_response.json()
    except Exception:
        return None

    return DrPrompt.from_dict(prompt_template_json)


def get_datarobot_prompt_template_version(
    prompt_template_id: str, prompt_template_version_id: str
) -> DrPromptVersion | None:
    api_client = get_api_client()
    try:
        prompt_template_version_response = api_client.get(
            f"genai/promptTemplates/{prompt_template_id}/versions/{prompt_template_version_id}/",
            join_endpoint=True,
        )
        prompt_template_version_json = prompt_template_version_response.json()
    except Exception:
        return None

    return DrPromptVersion.from_dict(prompt_template_version_json)
