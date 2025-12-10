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
from collections import defaultdict

import datarobot as dr

from datarobot_genai.drmcp.core.clients import get_api_client


def get_datarobot_prompt_templates() -> list[dr.genai.PromptTemplate]:
    try:
        return dr.genai.PromptTemplate.list()
    except Exception:
        return []


def get_datarobot_prompt_template_versions(
    prompt_template_ids: list[str],
) -> dict[str, list[dr.genai.PromptTemplateVersion]]:
    # Still missing in SDK
    prompt_template_versions_data = dr.utils.pagination.unpaginate(
        initial_url="genai/promptTemplates/versions/",
        initial_params={
            "promptTemplateIds": prompt_template_ids,
        },
        client=get_api_client(),
    )
    prompt_template_versions = defaultdict(list)
    for prompt_template_version in prompt_template_versions_data:
        prompt_template_versions[prompt_template_version["promptTemplateId"]].append(
            dr.genai.PromptTemplateVersion(
                id=prompt_template_version["id"],
                prompt_template_id=prompt_template_version["promptTemplateId"],
                prompt_text=prompt_template_version["promptText"],
                commit_comment=prompt_template_version["commitComment"],
                version=prompt_template_version["version"],
                variables=prompt_template_version["variables"],
                creation_date=prompt_template_version["creationDate"],
                creation_user_id=prompt_template_version["creationUserId"],
                user_name=prompt_template_version["userName"],
            )
        )
    return prompt_template_versions


def get_datarobot_prompt_template(prompt_template_id: str) -> dr.genai.PromptTemplate | None:
    try:
        return dr.genai.PromptTemplate.get(prompt_template_id)
    except Exception:
        return None


def get_datarobot_prompt_template_version(
    prompt_template_id: str, prompt_template_version_id: str
) -> dr.genai.PromptTemplateVersion | None:
    try:
        return dr.genai.PromptTemplateVersion.get(prompt_template_id, prompt_template_version_id)
    except Exception:
        return None
