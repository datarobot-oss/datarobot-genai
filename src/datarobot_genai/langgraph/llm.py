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

from collections.abc import AsyncIterator
from collections.abc import Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGenerationChunk  # noqa: TC002

from datarobot_genai.core.config import DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM
from datarobot_genai.core.config import Config
from datarobot_genai.core.config import LLMType
from datarobot_genai.core.config import default_api_key
from datarobot_genai.core.config import default_datarobot_llm_gateway_url
from datarobot_genai.core.config import default_deployment_url
from datarobot_genai.core.config import default_model_name


def _create_datarobot_chat_litellm(config: dict[str, Any]) -> Any:
    from langchain_litellm import ChatLiteLLM  # noqa: PLC0415

    if config.get("streaming"):
        config["stream_options"] = {"include_usage": True}
    else:
        config.pop("stream_options", None)

    return ChatLiteLLM(**config)


def get_datarobot_gateway_llm(
    model_name: str | None = None, parameters: dict | None = None, streaming: bool = True
) -> BaseChatModel:
    config = {
        "api_key": default_api_key(),
        "api_base": default_datarobot_llm_gateway_url(),
        "streaming": streaming,
    }

    if parameters:
        config.update(parameters)

    model_name = model_name or default_model_name()
    if model_name is None:
        raise ValueError("Model name is required")

    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config["model"] = model_name
    return _create_datarobot_chat_litellm(config)


def get_datarobot_deployment_llm(
    deployment_id: str,
    model_name: str | None = None,
    parameters: dict | None = None,
    streaming: bool = True,
) -> BaseChatModel:
    config = {
        "api_key": default_api_key(),
        "api_base": default_deployment_url(deployment_id),
        "streaming": streaming,
    }
    if parameters:
        config.update(parameters)

    model_name = model_name or default_model_name() or DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM
    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config["model"] = model_name
    return _create_datarobot_chat_litellm(config)


def get_datarobot_nim_llm(
    nim_deployment_id: str,
    model_name: str | None = None,
    parameters: dict | None = None,
    streaming: bool = True,
) -> BaseChatModel:
    config: dict[str, Any] = {
        "api_key": default_api_key(),
        "api_base": default_deployment_url(nim_deployment_id),
        "streaming": streaming,
    }
    if parameters:
        config.update(parameters)

    model_name = model_name or default_model_name()
    if model_name is None:
        raise ValueError("Model name is required")

    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config["model"] = model_name
    return _create_datarobot_chat_litellm(config)


def get_external_llm(
    model_name: str | None = None, parameters: dict | None = None, streaming: bool = True
) -> BaseChatModel:
    config: dict[str, Any] = {
        "streaming": streaming,
    }
    if parameters:
        config.update(parameters)

    model_name = model_name or default_model_name()
    if model_name is None:
        raise ValueError("Model name is required")

    model_name = model_name.removeprefix("datarobot/")

    config["model"] = model_name
    return _create_datarobot_chat_litellm(config)


def get_router_llm(
    primary: Any,
    fallbacks: list[Any],
    router_settings: dict | None = None,
) -> BaseChatModel:
    """Return a ``ChatLiteLLMRouter`` backed by a ``litellm.Router``.

    Args:
        primary: ``LLMConfig`` for the primary model.
        fallbacks: Ordered list of ``LLMConfig`` fallback configs.
        router_settings: Extra kwargs forwarded to ``litellm.Router``
            (e.g. ``allowed_fails``, ``cooldown_time``).
    """
    from langchain_litellm import ChatLiteLLMRouter  # noqa: PLC0415

    from datarobot_genai.core.router import build_litellm_router  # noqa: PLC0415

    class _CleanStreamRouter(ChatLiteLLMRouter):
        """Strip raw tool-call deltas from streaming ``additional_kwargs``.

        ``ChatLiteLLMRouter`` puts raw streaming tool-call delta objects into
        ``additional_kwargs["tool_calls"]``.  When chunks accumulate these become
        a flat list of partial deltas with fragmentary JSON arguments.  The
        correct data already lives in ``tool_call_chunks``; stripping the extra
        key lets downstream code use that path instead.
        """

        def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
            for chunk in super()._stream(*args, **kwargs):
                chunk.message.additional_kwargs.pop("tool_calls", None)
                yield chunk

        async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
            async for chunk in super()._astream(*args, **kwargs):
                chunk.message.additional_kwargs.pop("tool_calls", None)
                yield chunk

    router = build_litellm_router(primary, fallbacks, router_settings)
    return _CleanStreamRouter(router=router, model="primary", streaming=True)


def get_llm(
    model_name: str | None = None, parameters: dict | None = None, streaming: bool = True
) -> BaseChatModel:
    config = Config()
    llm_type = config.get_llm_type()
    if llm_type == LLMType.GATEWAY:
        return get_datarobot_gateway_llm(model_name, parameters, streaming)
    elif llm_type == LLMType.DEPLOYMENT:
        return get_datarobot_deployment_llm(
            config.llm_deployment_id,  # type: ignore[arg-type]
            model_name,
            parameters,
            streaming,
        )
    elif llm_type == LLMType.NIM:
        return get_datarobot_nim_llm(config.nim_deployment_id, model_name, parameters, streaming)  # type: ignore[arg-type]
    elif llm_type == LLMType.EXTERNAL:
        return get_external_llm(model_name, parameters, streaming)
    else:
        raise ValueError(f"Invalid LLM type inferred from config: {llm_type}, config: {config}")
