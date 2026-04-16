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

import json
from typing import Any

from crewai import LLM

from datarobot_genai.core.config import Config
from datarobot_genai.core.config import LLMType
from datarobot_genai.core.config import default_api_key
from datarobot_genai.core.config import default_datarobot_llm_gateway_url
from datarobot_genai.core.config import default_deployment_url
from datarobot_genai.core.config import default_model_name


def _crewai_model_factory(config: dict) -> LLM:
    config["stream_options"] = config.get("stream_options", {"include_usage": True})

    # This class is used to override all the magic LLM tries to pull on using
    # native LLM clients. We don't want to use native LLM clients, we want to use
    # LiteLLM for the way we establish our agents.
    class LitellmOnlyLLM(LLM):
        def __new__(cls, *args: Any, **kwargs: Any) -> "LitellmOnlyLLM":
            return object.__new__(cls)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.is_litellm = True

    return LitellmOnlyLLM(**config)


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


def _serialize_router_tool_calls(message: Any) -> str | None:
    """Serialize tool_calls from a litellm message to JSON for CrewAI parsing.

    CrewAI expects tool calls to be JSON-serialized within the content field.
    Returns None if no tool calls present; otherwise returns JSON string.
    """
    if not getattr(message, "tool_calls", None):
        return None
    return json.dumps(
        {
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        }
    )


def get_router_llm(
    primary_config: Any,
    fallback_configs: list[Any],
    router_settings: dict | None = None,
) -> LLM:
    """Return a CrewAI ``LLM`` whose calls are routed through a ``litellm.Router``.

    Args:
        primary_config: ``DataRobotLLMComponentModelConfig`` for the primary model.
        fallback_configs: Ordered list of fallback configs.
        router_settings: Extra kwargs forwarded to ``litellm.Router``.
    """
    from datarobot_genai.core.router import _config_to_litellm_params
    from datarobot_genai.core.router import build_litellm_router

    router = build_litellm_router(
        _config_to_litellm_params(primary_config),
        [_config_to_litellm_params(c) for c in fallback_configs],
        router_settings,
    )

    class RouterLitellmOnlyLLM(LLM):
        def __new__(cls, *args: Any, **kwargs: Any) -> "RouterLitellmOnlyLLM":
            return object.__new__(cls)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.is_litellm = True
            self._llm_router = router

        def call(
            self,
            messages: list[dict],
            tools: list[dict] | None = None,
            callbacks: list | None = None,
            available_tools: list[dict] | None = None,
            **kwargs: Any,
        ) -> str:
            resp = self._llm_router.completion(
                "primary",
                messages=messages,
                **({"tools": tools} if tools else {}),
            )
            message = resp.choices[0].message
            content = message.content or ""
            tool_calls_json = _serialize_router_tool_calls(message)
            return tool_calls_json or content

        async def acall(
            self,
            messages: list[dict],
            tools: list[dict] | None = None,
            callbacks: list | None = None,
            available_tools: list[dict] | None = None,
            **kwargs: Any,
        ) -> str:
            resp = await self._llm_router.acompletion(
                "primary",
                messages=messages,
                **({"tools": tools} if tools else {}),
            )
            message = resp.choices[0].message
            content = message.content or ""
            tool_calls_json = _serialize_router_tool_calls(message)
            return tool_calls_json or content

    return RouterLitellmOnlyLLM(model="datarobot-router")


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
