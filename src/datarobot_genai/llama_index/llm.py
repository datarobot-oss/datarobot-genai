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

from llama_index.llms.litellm import LiteLLM

from datarobot_genai.core.config import Config
from datarobot_genai.core.config import LLMType
from datarobot_genai.core.config import default_api_key
from datarobot_genai.core.config import default_datarobot_llm_gateway_url
from datarobot_genai.core.config import default_deployment_url
from datarobot_genai.core.config import default_model_name


def _create_datarobot_litellm(config: dict[str, Any]) -> Any:
    from llama_index.core.base.llms.types import LLMMetadata  # noqa: PLC0415
    from llama_index.llms.litellm import LiteLLM  # noqa: PLC0415

    class DataRobotLiteLLM(LiteLLM):  # type: ignore[misc]
        """DataRobotLiteLLM is a small LiteLLM wrapper class that makes all LiteLLM endpoints
        compatible with the LlamaIndex library.
        """

        @property
        def metadata(self) -> LLMMetadata:
            """Returns the metadata for the LLM.

            This is required to enable the is_chat_model and is_function_calling_model, which are
            mandatory for LlamaIndex agents. By default, LlamaIndex assumes these are false unless
            each individual model config in LiteLLM explicitly sets them to true. To use custom LLM
            endpoints with LlamaIndex agents, you must override this method to return the
            appropriate metadata.
            """
            return LLMMetadata(
                context_window=128000,
                num_output=self.max_tokens or -1,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
            )

    return DataRobotLiteLLM(**config)


def get_datarobot_gateway_llm(
    model_name: str | None = None, parameters: dict | None = None
) -> LiteLLM:
    config = {
        "api_key": default_api_key(),
        "api_base": default_datarobot_llm_gateway_url(),
        "stream_options": {"include_usage": True},
    }

    if parameters:
        config.update(parameters)

    model_name = model_name or default_model_name()

    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config["model"] = model_name
    return _create_datarobot_litellm(config)


def get_datarobot_deployment_llm(
    deployment_id: str, model_name: str | None = None, parameters: dict | None = None
) -> LiteLLM:
    config = {
        "api_key": default_api_key(),
        "api_base": default_deployment_url(deployment_id),
        "stream_options": {"include_usage": True},
    }

    if parameters:
        config.update(parameters)

    model_name = model_name or default_model_name()
    if not model_name.startswith("datarobot/"):
        model_name = "datarobot/" + model_name

    config["model"] = model_name
    return _create_datarobot_litellm(config)


def get_datarobot_nim_llm(
    nim_deployment_id: str, model_name: str | None = None, parameters: dict | None = None
) -> LiteLLM:
    return get_datarobot_deployment_llm(nim_deployment_id, model_name, parameters)


def get_external_llm(model_name: str | None = None, parameters: dict | None = None) -> LiteLLM:
    config = {
        # Everything else is loaded from the environment by LiteLLM
    }
    if parameters:
        config.update(parameters)

    model_name = model_name or default_model_name()
    model_name = model_name.removeprefix("datarobot/")
    config["model"] = model_name
    return _create_datarobot_litellm(config)


def get_router_llm(
    primary: Any,
    fallbacks: list[Any],
    router_settings: dict | None = None,
) -> LiteLLM:
    """Return a LlamaIndex ``LiteLLM`` whose calls are routed through a ``litellm.Router``.

    Args:
        primary: ``LLMConfig`` for the primary model.
        fallbacks: Ordered list of ``LLMConfig`` fallback configs.
        router_settings: Extra kwargs forwarded to ``litellm.Router``.
    """
    from llama_index.core.base.llms.types import LLMMetadata  # noqa: PLC0415

    from datarobot_genai.core.router import build_litellm_router  # noqa: PLC0415

    router = build_litellm_router(primary, fallbacks, router_settings)

    class RouterDataRobotLiteLLM(LiteLLM):  # type: ignore[misc]
        """LlamaIndex LiteLLM subclass that delegates to a litellm.Router with streaming."""

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                context_window=128000,
                num_output=self.max_tokens or -1,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
            )

        def _chat(self, messages: Any, **kwargs: Any) -> Any:
            from llama_index.core.base.llms.types import ChatMessage  # noqa: PLC0415
            from llama_index.core.base.llms.types import ChatResponse  # noqa: PLC0415
            from llama_index.llms.litellm.utils import to_openai_message_dicts  # noqa: PLC0415

            message_dicts = to_openai_message_dicts(messages)
            resp = router.completion("primary", messages=message_dicts, **kwargs)
            message = resp.choices[0].message
            content = message.content or ""
            additional_kwargs: dict = {}
            if message.tool_calls:
                additional_kwargs["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]
            return ChatResponse(
                message=ChatMessage(
                    role="assistant", content=content, additional_kwargs=additional_kwargs
                ),
                raw=resp,
            )

        async def _achat(self, messages: Any, **kwargs: Any) -> Any:
            from llama_index.core.base.llms.types import ChatMessage  # noqa: PLC0415
            from llama_index.core.base.llms.types import ChatResponse  # noqa: PLC0415
            from llama_index.llms.litellm.utils import to_openai_message_dicts  # noqa: PLC0415

            message_dicts = to_openai_message_dicts(messages)
            resp = await router.acompletion("primary", messages=message_dicts, **kwargs)
            message = resp.choices[0].message
            content = message.content or ""
            additional_kwargs: dict = {}
            if message.tool_calls:
                additional_kwargs["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]
            return ChatResponse(
                message=ChatMessage(
                    role="assistant", content=content, additional_kwargs=additional_kwargs
                ),
                raw=resp,
            )

        def _stream_chat(self, messages: Any, **kwargs: Any) -> Any:
            from llama_index.core.base.llms.types import ChatMessage  # noqa: PLC0415
            from llama_index.core.base.llms.types import ChatResponse  # noqa: PLC0415
            from llama_index.llms.litellm.utils import to_openai_message_dicts  # noqa: PLC0415
            from llama_index.llms.litellm.utils import update_tool_calls  # noqa: PLC0415

            message_dicts = to_openai_message_dicts(messages)
            accumulated: list[str] = []
            tool_calls: list[dict] = []
            for chunk in router.completion(
                "primary", messages=message_dicts, stream=True, **kwargs
            ):
                delta = chunk.choices[0].delta
                content = delta.content or ""
                if content:
                    accumulated.append(content)
                tool_call_delta = getattr(delta, "tool_calls", None)
                if tool_call_delta:
                    tool_calls = update_tool_calls(tool_calls, tool_call_delta)
                additional_kwargs: dict = {}
                if tool_calls:
                    additional_kwargs["tool_calls"] = tool_calls
                yield ChatResponse(
                    message=ChatMessage(
                        role="assistant",
                        content="".join(accumulated),
                        additional_kwargs=additional_kwargs,
                    ),
                    delta=content,
                    raw=chunk,
                )

        async def _astream_chat(self, messages: Any, **kwargs: Any) -> Any:
            from llama_index.core.base.llms.types import ChatMessage  # noqa: PLC0415
            from llama_index.core.base.llms.types import ChatResponse  # noqa: PLC0415
            from llama_index.llms.litellm.utils import to_openai_message_dicts  # noqa: PLC0415
            from llama_index.llms.litellm.utils import update_tool_calls  # noqa: PLC0415

            message_dicts = to_openai_message_dicts(messages)

            async def gen() -> Any:
                accumulated: list[str] = []
                tool_calls: list[dict] = []
                async for chunk in await router.acompletion(
                    "primary", messages=message_dicts, stream=True, **kwargs
                ):
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                    if content:
                        accumulated.append(content)
                    tool_call_delta = getattr(delta, "tool_calls", None)
                    if tool_call_delta:
                        tool_calls = update_tool_calls(tool_calls, tool_call_delta)
                    additional_kwargs: dict = {}
                    if tool_calls:
                        additional_kwargs["tool_calls"] = tool_calls
                    yield ChatResponse(
                        message=ChatMessage(
                            role="assistant",
                            content="".join(accumulated),
                            additional_kwargs=additional_kwargs,
                        ),
                        delta=content,
                        raw=chunk,
                    )

            return gen()

    return RouterDataRobotLiteLLM(model="primary")


def get_llm(model_name: str | None = None, parameters: dict | None = None) -> LiteLLM:
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
