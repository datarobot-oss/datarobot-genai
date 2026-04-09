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

from typing import Any

from langchain_core.language_models import BaseChatModel

from datarobot_genai.core.config import default_api_key
from datarobot_genai.core.config import default_datarobot_llm_gateway_url
from datarobot_genai.core.config import default_deployment_url
from datarobot_genai.core.config import default_model_name


def get_datarobot_gateway_llm(
    model_name: str | None = None, streaming: bool = True, parameters: dict | None = None
) -> BaseChatModel:
    model_name = model_name or default_model_name()
    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config = {
        "model": model_name,
        "api_key": default_api_key(),
        "api_base": default_datarobot_llm_gateway_url(),
        "streaming": streaming,
        "stream_options": {"include_usage": True},
    }
    if parameters:
        config.update(parameters)
    return _create_datarobot_chat_openai(config)


def get_datarobot_deployment_llm(
    deployment_id: str,
    model_name: str | None = None,
    streaming: bool = True,
    parameters: dict | None = None,
) -> BaseChatModel:
    model_name = model_name or default_model_name()
    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config = {
        "model": model_name,
        "api_key": default_api_key(),
        "api_base": default_deployment_url(deployment_id),
        "streaming": streaming,
        "stream_options": {"include_usage": True},
    }
    if parameters:
        config.update(parameters)
    return _create_datarobot_chat_openai(config)


def get_datarobot_nim_llm(
    nim_deployment_id: str, model_name: str | None = None, parameters: dict | None = None
) -> BaseChatModel:
    return get_datarobot_deployment_llm(nim_deployment_id, model_name, parameters)


def _create_datarobot_chat_openai(config: dict[str, Any]) -> Any:
    from langchain_litellm import ChatLiteLLM  # noqa: PLC0415

    class DataRobotChatLiteLLM(ChatLiteLLM):
        def _get_request_payload(  # type: ignore[override]
            self,
            *args: Any,
            **kwargs: Any,
        ) -> dict:
            # We need to default to include_usage=True for streaming but we get 400 response
            # if stream_options is present for a non-streaming call.
            payload = super()._get_request_payload(*args, **kwargs)
            if not payload.get("streaming"):
                payload.pop("stream_options", None)
            return payload

    return DataRobotChatLiteLLM(**config)
