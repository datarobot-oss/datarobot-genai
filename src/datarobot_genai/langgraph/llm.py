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
from typing import Any

from datarobot_genai.core.config import Config
from datarobot_genai.nat.helpers import extract_headers_from_context
from nat.plugins.langchain.llm import (_patch_llm_based_on_config as langchain_patch_llm_based_on_config)

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


def _create_datarobot_chat_openai(config: dict[str, Any]) -> Any:
    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    class DataRobotChatOpenAI(ChatOpenAI):
        def _get_request_payload(  # type: ignore[override]
            self,
            *args: Any,
            **kwargs: Any,
        ) -> dict:
            # We need to default to include_usage=True for streaming but we get 400 response
            # if stream_options is present for a non-streaming call.
            payload = super()._get_request_payload(*args, **kwargs)
            if not payload.get("stream"):
                payload.pop("stream_options", None)
            return payload

    return DataRobotChatOpenAI(**config)


async def get_gateway_llm(llm_name: str) -> ChatOpenAI:
    llm_config = Config(llm_default_model=llm_name, streamming=True, use_datarobot_llm_gateway=True)

    config = {
        "base_url": llm_config.default_base_url() + "/genai/llmgw",
        "stream_options": {"include_usage": True},
        "model": llm_config.llm_default_model.removeprefix("datarobot/"),  #TODO check if needed
        "api_key": llm_config.default_api_key(),
    }
    client = _create_datarobot_chat_openai(config)
    yield langchain_patch_llm_based_on_config(client, config)


async def get_deployment_llm(deployment_id: str) -> ChatOpenAI:
    # from Config base new Config() 
    # fetch data from this config stream: parametr false api key and token
    llm_config = Config(llm_deployment_id=deployment_id, streamming=True)
    config = {
        "base_url": llm_config.default_base_url(),
        "stream_options": {"include_usage": True},
        "model": llm_config.default_model_name().removeprefix("datarobot/"), #TODO check if needed 
        "api_key": llm_config.default_api_key()
    }
    # TODO do we need headers?
    context_headers = extract_headers_from_context(["X-DataRobot-Identity-Token"])
    if llm_config.headers:
        context_headers = {**context_headers, **llm_config.headers}

    config["default_headers"] = context_headers

    client = _create_datarobot_chat_openai(config)
    yield langchain_patch_llm_based_on_config(client, config)
