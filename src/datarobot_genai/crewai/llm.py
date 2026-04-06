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

from __future__ import annotations

from typing import TYPE_CHECKING

from datarobot_genai.core.config import Config
from datarobot_genai.core.utils.llm import _patch_llm_based_on_config

if TYPE_CHECKING:
    from crewai import LLM


def get_gateway_llm(llm_name: str) -> LLM:
    from crewai import LLM  # noqa: PLC0415
    llm_config = Config(llm_default_model=llm_name, streamming=True, use_datarobot_llm_gateway=True)
    config = {
        "api_key": llm_config.default_api_key(),
    }
    if not llm_config.llm_default_model.startswith("datarobot/"):
        config["model"] = "datarobot/" + llm_config.llm_default_model
    config["base_url"] = llm_config.default_base_url().removesuffix("/api/v2")
    client = LLM(**config)
    return _patch_llm_based_on_config(client, llm_config)


def get_deployment_llm(deployment_id: str) -> LLM:
    from crewai import LLM  # noqa: PLC0415

    llm_config = Config(llm_deployment_id=deployment_id, streamming=True)
    config = {
        "base_url": llm_config.default_base_url(),
        "api_key": llm_config.default_api_key()
    }

    if not llm_config.llm_default_model.startswith("datarobot/"):
        config["model"] = "datarobot/" + llm_config.llm_default_model
    config["api_base"] = config.pop("base_url") + "/chat/completions"
    # TODO do we need headers?
    if llm_config.headers:
        config["extra_headers"] = llm_config.headers

    client = LLM(**config)
    return _patch_llm_based_on_config(client, llm_config)
