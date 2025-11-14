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
import datarobot as dr

from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import get_datarobot_prompt_template_versions
from datarobot_genai.drmcp.core.dynamic_prompts.dr_lib import get_datarobot_prompt_templates


def get_or_create_prompt_template(name: str) -> dict:
    try:
        for prompt_template in get_datarobot_prompt_templates():
            if prompt_template.name == name:
                return {
                    "id": prompt_template.id,
                    "name": name,
                }
    except Exception as e:
        print(f"Error checking for existing prompt template: {e}")

    try:
        client = dr.client.get_client()
        r = client.post(
            url="genai/promptTemplates/",
            data={
                "name": name,
                "description": f"Description for {name}",
            },
            join_endpoint=True,
        )
        return {
            "id": r.json()["id"],
            "name": name,
        }
    except Exception as e:
        print(f"Error creating prompt template: {e}")
        raise


def get_or_create_prompt_template_version(
    prompt_template_id: str, prompt_text: str, variables: list[str]
) -> dict:
    try:
        for prompt_template_version in get_datarobot_prompt_template_versions(prompt_template_id):
            if prompt_template_version.prompt_text == prompt_text:
                return {
                    "id": prompt_template_version.id,
                    "prompt_text": prompt_template_version.prompt_text,
                }
    except Exception as e:
        print(f"Error checking for existing prompt template versions: {e}")

    try:
        client = dr.client.get_client()
        r = client.post(
            url=f"genai/promptTemplates/{prompt_template_id}/versions/",
            data={
                "promptText": prompt_text,
                "commitComment": "Dummy commit comment",
                "variables": [
                    {
                        "name": v,
                        "description": f"Description for {v}",
                        "type": "str",
                    }
                    for v in variables
                ],
            },
            join_endpoint=True,
        )
        return {"id": r.json()["id"], "prompt_text": prompt_text}
    except Exception as e:
        print(f"Error creating prompt template version: {e}")
        raise
