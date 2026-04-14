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

from datarobot_genai.core.config import Config
from datarobot_genai.core.config import LLMType
from datarobot_genai.core.config import default_api_key
from datarobot_genai.core.config import default_datarobot_llm_gateway_url
from datarobot_genai.core.config import default_deployment_url
from datarobot_genai.core.config import default_model_name


def _crewai_model_factory(config: dict) -> LLM:
    config["stream_options"] = config.get("stream_options", {"include_usage": True})
    llm = LLM(**config, is_litellm=True)
    llm.additional_params.pop("is_litellm", None)
    return llm


def get_datarobot_gateway_llm(model_name: str | None = None, parameters: dict | None = None) -> LLM:
    config = {
        "api_key": default_api_key(),
        "api_base": default_datarobot_llm_gateway_url(),
    }

    if parameters:
        config.update(parameters)

    model_name = model_name or default_model_name()
    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config["model"] = model_name

    return _crewai_model_factory(config)


def get_datarobot_deployment_llm(
    deployment_id: str, model_name: str | None = None, parameters: dict | None = None
) -> LLM:
    config = {
        "api_key": default_api_key(),
        "api_base": default_deployment_url(deployment_id),
    }

    if parameters:
        config.update(parameters)

    model_name = model_name or default_model_name()
    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config["model"] = model_name
    return _crewai_model_factory(config)


def get_datarobot_nim_llm(
    nim_deployment_id: str, model_name: str | None = None, parameters: dict | None = None
) -> LLM:
    return get_datarobot_deployment_llm(nim_deployment_id, model_name, parameters)


def get_external_llm(model_name: str | None = None, parameters: dict | None = None) -> LLM:
    config = {
        # Everything else is loaded from the environment by LiteLLM
    }

    if parameters:
        config.update(parameters)
    model_name = model_name or default_model_name()
    model_name = model_name.removeprefix("datarobot/")
    config["model"] = model_name

    return _crewai_model_factory(config)


def get_llm(model_name: str | None = None, parameters: dict | None = None) -> LLM:
    config = Config()
    llm_type = config.get_llm_type()
    if llm_type == LLMType.GATEWAY:
        return get_datarobot_gateway_llm(model_name, parameters)
    elif llm_type == LLMType.DEPLOYMENT:
        return get_datarobot_deployment_llm(config.llm_deployment_id, model_name, parameters)  # type: ignore[arg-type]
    elif llm_type == LLMType.NIM:
        return get_datarobot_nim_llm(config.nim_deployment_id, model_name, parameters)  # type: ignore[arg-type]
    elif llm_type == LLMType.EXTERNAL:
        return get_external_llm(model_name, parameters)
    else:
        raise ValueError(f"Invalid LLM type inferred from config: {llm_type}, config: {config}")
