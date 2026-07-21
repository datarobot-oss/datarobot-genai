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
from langchain_core.messages import BaseMessage  # noqa: TC002
from langchain_core.outputs import ChatGenerationChunk  # noqa: TC002

from datarobot_genai.core.config import DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM
from datarobot_genai.core.config import LLMConfig
from datarobot_genai.core.config import LLMType
from datarobot_genai.core.config import default_api_key
from datarobot_genai.core.config import default_datarobot_llm_gateway_url
from datarobot_genai.core.config import default_deployment_url
from datarobot_genai.core.config import default_model_name
from datarobot_genai.core.config import resolve_config
from datarobot_genai.core.llm_parameters import apply_reasoning_to_parameters


def _wrap_bare_text_blocks(
    message_dicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Wrap bare-string items in list-form content as ``{"type": "text"}`` blocks.

    When ``langchain_litellm`` serializes a streamed reasoning-model turn it drops
    the thinking blocks but leaves the answer as a bare string inside the content
    list. litellm's OpenAI request transform then calls ``item.get("type")`` on
    every list item and raises ``'str' object has no attribute 'get'`` on that
    string. Wrapping each bare string as a text block makes the content valid for
    the wire; dict items (multimodal ``image_url``/``file`` parts, etc.) and
    plain-string content are left untouched. Empty-string fragments that langchain
    leaves between blocks are dropped, and an all-empty list collapses back to
    ``""`` so the wire never carries an empty content array.
    """
    for message_dict in message_dicts:
        content = message_dict.get("content")
        if isinstance(content, list):
            message_dict["content"] = [
                {"type": "text", "text": item} if isinstance(item, str) else item
                for item in content
                if item != ""
            ] or ""
    return message_dicts


def _create_datarobot_chat_litellm(config: dict[str, Any]) -> Any:
    from langchain_litellm import ChatLiteLLM  # noqa: PLC0415

    if config.get("streaming"):
        config["stream_options"] = {"include_usage": True}
    else:
        config.pop("stream_options", None)

    extra_body = config.pop("extra_body", None)
    if extra_body is not None:
        model_kwargs = config.get("model_kwargs") or {}
        model_kwargs["extra_body"] = extra_body
        config["model_kwargs"] = model_kwargs

    class _ChatLiteLLM(ChatLiteLLM):  # type: ignore[valid-type,misc]
        """ChatLiteLLM that keeps list-form content valid for litellm's request transform.

        All four generation paths (``_generate``/``_agenerate``/``_stream``/
        ``_astream``) funnel through ``_create_message_dicts`` to build the wire
        request, so overriding that one method covers them all. See
        ``_wrap_bare_text_blocks``.
        """

        def _create_message_dicts(
            self, messages: list[BaseMessage], stop: list[str] | None
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            message_dicts, params = super()._create_message_dicts(messages, stop)
            return _wrap_bare_text_blocks(message_dicts), params

    return _ChatLiteLLM(**config)


def get_datarobot_gateway_llm(
    model_name: str | None = None,
    parameters: dict | None = None,
    streaming: bool = True,
    reasoning: bool = False,
) -> BaseChatModel:
    config = {
        "api_key": default_api_key(),
        "api_base": default_datarobot_llm_gateway_url(),
        "streaming": streaming,
    }

    model_name = model_name or default_model_name()
    if model_name is None:
        raise ValueError("Model name is required")

    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config.update(
        apply_reasoning_to_parameters(parameters, reasoning=reasoning, model_name=model_name)
    )
    config["model"] = model_name
    return _create_datarobot_chat_litellm(config)


def get_datarobot_deployment_llm(
    deployment_id: str,
    model_name: str | None = None,
    parameters: dict | None = None,
    streaming: bool = True,
    reasoning: bool = False,
) -> BaseChatModel:
    config = {
        "api_key": default_api_key(),
        "api_base": default_deployment_url(deployment_id),
        "streaming": streaming,
    }

    model_name = model_name or default_model_name() or DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM
    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config.update(
        apply_reasoning_to_parameters(parameters, reasoning=reasoning, model_name=model_name)
    )
    config["model"] = model_name
    return _create_datarobot_chat_litellm(config)


def get_datarobot_nim_llm(
    nim_deployment_id: str,
    model_name: str | None = None,
    parameters: dict | None = None,
    streaming: bool = True,
    reasoning: bool = False,
) -> BaseChatModel:
    config: dict[str, Any] = {
        "api_key": default_api_key(),
        "api_base": default_deployment_url(nim_deployment_id),
        "streaming": streaming,
    }

    model_name = model_name or default_model_name()
    if model_name is None:
        raise ValueError("Model name is required")

    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config.update(
        apply_reasoning_to_parameters(parameters, reasoning=reasoning, model_name=model_name)
    )
    config["model"] = model_name
    return _create_datarobot_chat_litellm(config)


def get_external_llm(
    model_name: str | None = None,
    parameters: dict | None = None,
    streaming: bool = True,
    reasoning: bool = False,
) -> BaseChatModel:
    config: dict[str, Any] = {
        "streaming": streaming,
    }

    model_name = model_name or default_model_name()
    if model_name is None:
        raise ValueError("Model name is required")

    model_name = model_name.removeprefix("datarobot/")

    config.update(
        apply_reasoning_to_parameters(parameters, reasoning=reasoning, model_name=model_name)
    )
    config["model"] = model_name
    return _create_datarobot_chat_litellm(config)


def get_router_llm(
    primary: LLMConfig,
    fallbacks: list[LLMConfig],
    router_settings: dict | None = None,
) -> BaseChatModel:
    """Return a ``ChatLiteLLMRouter`` backed by a ``litellm.Router``.

    Args:
        primary: ``LLMConfig`` for the primary model.
        fallbacks: Ordered list of ``LLMConfig`` fallback configs.
        router_settings: Extra kwargs forwarded to ``litellm.Router``
            (e.g. ``num_retries``).
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

        Also normalizes list-form content (see ``_wrap_bare_text_blocks``) so the
        router path gets the same protection as the non-router model.
        """

        def _create_message_dicts(
            self, messages: list[BaseMessage], stop: list[str] | None
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            message_dicts, params = super()._create_message_dicts(messages, stop)
            return _wrap_bare_text_blocks(message_dicts), params

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
    model_name: str | None = None,
    parameters: dict | None = None,
    streaming: bool = True,
    reasoning: bool = False,
) -> BaseChatModel:
    config = resolve_config()
    llm_type = config.get_llm_type()
    if llm_type == LLMType.GATEWAY:
        return get_datarobot_gateway_llm(model_name, parameters, streaming, reasoning)
    elif llm_type == LLMType.DEPLOYMENT:
        return get_datarobot_deployment_llm(
            config.llm_deployment_id,  # type: ignore[arg-type]
            model_name,
            parameters,
            streaming,
            reasoning,
        )
    elif llm_type == LLMType.NIM:
        return get_datarobot_nim_llm(
            config.nim_deployment_id,  # type: ignore[arg-type]
            model_name,
            parameters,
            streaming,
            reasoning,
        )
    elif llm_type == LLMType.EXTERNAL:
        return get_external_llm(model_name, parameters, streaming, reasoning)
    else:
        raise ValueError(f"Invalid LLM type inferred from config: {llm_type}, config: {config}")
