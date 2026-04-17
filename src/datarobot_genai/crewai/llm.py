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
    # Strip NAT-internal keys that cause "extra inputs" errors in litellm.
    # Multiple config types (Deployment, Component, Litellm) flow through here.
    config.pop("verify_ssl", None)

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
    """Serialize tool_calls from a litellm message to JSON for CrewAI parsing."""
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
    primary: Any,
    fallbacks: list[Any],
    router_settings: dict | None = None,
) -> LLM:
    """Return a CrewAI ``LLM`` whose calls are routed through a ``litellm.Router``.

    Args:
        primary: ``LLMConfig`` for the primary model.
        fallbacks: Ordered list of ``LLMConfig`` fallback configs.
        router_settings: Extra kwargs forwarded to ``litellm.Router``.
    """
    from datarobot_genai.core.router import build_litellm_router  # noqa: PLC0415

    router = build_litellm_router(primary, fallbacks, router_settings)

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
            accumulated = []
            tool_calls_seen: list[Any] = []
            for chunk in self._llm_router.completion(
                "primary",
                messages=messages,
                stream=True,
                **({"tools": tools} if tools else {}),
            ):
                delta = chunk.choices[0].delta
                if delta.content:
                    accumulated.append(delta.content)
                    if callbacks:
                        for cb in callbacks:
                            if hasattr(cb, "on_llm_new_token"):
                                cb.on_llm_new_token(delta.content)
                if getattr(delta, "tool_calls", None):
                    tool_calls_seen.extend(delta.tool_calls)
            if tool_calls_seen:
                # Reconstruct complete tool calls from streaming deltas
                merged: dict[int, dict] = {}
                for tc in tool_calls_seen:
                    idx = tc.index
                    if idx not in merged:
                        merged[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        merged[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        merged[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        merged[idx]["arguments"] += tc.function.arguments
                return json.dumps(
                    {
                        "tool_calls": [
                            {
                                "id": v["id"],
                                "type": "function",
                                "function": {"name": v["name"], "arguments": v["arguments"]},
                            }
                            for v in merged.values()
                        ]
                    }
                )
            return "".join(accumulated)

        async def acall(
            self,
            messages: list[dict],
            tools: list[dict] | None = None,
            callbacks: list | None = None,
            available_tools: list[dict] | None = None,
            **kwargs: Any,
        ) -> str:
            accumulated = []
            tool_calls_seen: list[Any] = []
            response = await self._llm_router.acompletion(
                "primary",
                messages=messages,
                stream=True,
                **({"tools": tools} if tools else {}),
            )
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    accumulated.append(delta.content)
                    if callbacks:
                        for cb in callbacks:
                            if hasattr(cb, "on_llm_new_token"):
                                cb.on_llm_new_token(delta.content)
                if getattr(delta, "tool_calls", None):
                    tool_calls_seen.extend(delta.tool_calls)
            if tool_calls_seen:
                merged: dict[int, dict] = {}
                for tc in tool_calls_seen:
                    idx = tc.index
                    if idx not in merged:
                        merged[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        merged[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        merged[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        merged[idx]["arguments"] += tc.function.arguments
                return json.dumps(
                    {
                        "tool_calls": [
                            {
                                "id": v["id"],
                                "type": "function",
                                "function": {"name": v["name"], "arguments": v["arguments"]},
                            }
                            for v in merged.values()
                        ]
                    }
                )
            return "".join(accumulated)

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
