# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from crewai import LLM

from datarobot_genai.core.config import default_api_key
from datarobot_genai.core.config import default_datarobot_llm_gateway_url
from datarobot_genai.core.config import default_deployment_url
from datarobot_genai.core.config import default_model_name


def get_datarobot_gateway_llm(
    model_name: str | None = None, parameters: dict | None = None
) -> LLM:
    model_name = model_name or default_model_name()
    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config = {
        "model": model_name,
        "api_key": default_api_key(),
        "api_base": default_datarobot_llm_gateway_url(),
        "stream_options": {"include_usage": True},
    }

    config["api_base"] = config["api_base"].removesuffix("/api/v2")
    if parameters:
        config.update(parameters)
    return LLM(**config)

def get_datarobot_deployment_llm(
    deployment_id: str, model_name: str | None = None, parameters: dict | None = None
) -> LLM:
    model_name = model_name or default_model_name()
    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config = {
        "model": model_name,
        "api_key": default_api_key(),
        "api_base": default_deployment_url(deployment_id),
        "stream_options": {"include_usage": True},
    }

    config["api_base"] = config["api_base"] + "/chat/completions"
    if parameters:
        config.update(parameters)
    return LLM(**config)

def get_datarobot_nim_llm(
    nim_deployment_id: str, model_name: str | None = None, parameters: dict | None = None
) -> LLM:
    return get_datarobot_deployment_llm(nim_deployment_id, model_name, parameters)