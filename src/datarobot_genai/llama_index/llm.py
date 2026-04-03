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

"""Construct LlamaIndex LiteLLM clients for DataRobot gateway and deployments."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.utils.responses_api import validate_no_responses_api

from datarobot_genai.nat.datarobot_llm_clients import _create_datarobot_litellm
from datarobot_genai.nat.datarobot_llm_clients import _patch_llm_based_on_config
from datarobot_genai.nat.datarobot_llm_providers import Config
from datarobot_genai.nat.datarobot_llm_providers import DataRobotLLMDeploymentModelConfig
from datarobot_genai.nat.datarobot_llm_providers import DataRobotLLMGatewayModelConfig

if TYPE_CHECKING:
    from llama_index.llms.litellm import LiteLLM


def get_gateway_llm(llm_name: str) -> LiteLLM:
    llm_config = DataRobotLLMGatewayModelConfig(model=llm_name)
    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)
    config = llm_config.model_dump(
        exclude={"type", "thinking", "api_type"},
        by_alias=True,
        exclude_none=True,
    )
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url").removesuffix("/api/v2")
    client = _create_datarobot_litellm(config)
    return _patch_llm_based_on_config(client, llm_config)


def get_deployment_llm(deployment_id: str) -> LiteLLM:
    dr = Config()
    base = dr.datarobot_endpoint.rstrip("/")
    base_url = f"{base}/deployments/{deployment_id}"
    llm_config = DataRobotLLMDeploymentModelConfig(base_url=base_url)
    validate_no_responses_api(llm_config, LLMFrameworkEnum.LLAMA_INDEX)
    config = llm_config.model_dump(
        exclude={"type", "thinking", "headers", "api_type"},
        by_alias=True,
        exclude_none=True,
    )
    if not config["model"].startswith("datarobot/"):
        config["model"] = "datarobot/" + config["model"]
    config["api_base"] = config.pop("base_url") + "/chat/completions"
    if llm_config.headers:
        config["additional_kwargs"] = {"extra_headers": llm_config.headers}

    client = _create_datarobot_litellm(config)
    return _patch_llm_based_on_config(client, llm_config)
