# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Map a LangChain chat model from NAT's ``EvalBuilder`` to a litellm target."""

from __future__ import annotations

from typing import Any
from typing import cast


def _secret_value(value: object) -> str | None:
    if value is None:
        return None
    getter = getattr(value, "get_secret_value", None)
    return getter() if callable(getter) else str(value)


def _litellm_kwargs_from_chat_litellm(llm: object) -> dict[str, Any]:
    from langchain_litellm import ChatLiteLLM

    if not isinstance(llm, ChatLiteLLM):
        raise TypeError("expected ChatLiteLLM")

    completion_kwargs: dict[str, Any] = {}
    api_key = _secret_value(getattr(llm, "api_key", None))
    if api_key:
        completion_kwargs["api_key"] = api_key
    api_base = getattr(llm, "api_base", None)
    if api_base:
        completion_kwargs["api_base"] = api_base
    extra_headers = getattr(llm, "extra_headers", None)
    if extra_headers:
        completion_kwargs["extra_headers"] = dict(extra_headers)
    model_kwargs = getattr(llm, "model_kwargs", None) or {}
    if isinstance(model_kwargs, dict) and model_kwargs.get("extra_body") is not None:
        completion_kwargs["extra_body"] = model_kwargs["extra_body"]
    return completion_kwargs


def langchain_chat_model_to_litellm(llm: object) -> tuple[str, dict[str, Any]]:
    """Resolve ``litellm.acompletion`` ``model`` and kwargs from a LangChain chat model."""
    from langchain_litellm import ChatLiteLLM
    from langchain_openai import AzureChatOpenAI
    from langchain_openai import ChatOpenAI

    if isinstance(llm, ChatLiteLLM):
        model = getattr(llm, "model", None)
        if not model:
            raise ValueError("ChatLiteLLM client has no model name configured.")
        return str(model), _litellm_kwargs_from_chat_litellm(llm)

    if isinstance(llm, AzureChatOpenAI):
        deployment = llm.deployment_name or llm.model_name
        if not deployment:
            raise ValueError(
                "Could not determine Azure deployment name from AzureChatOpenAI client."
            )
        return f"azure/{deployment}", {
            "api_key": _secret_value(llm.openai_api_key),
            "api_base": llm.azure_endpoint,
            "api_version": llm.openai_api_version,
        }

    if isinstance(llm, ChatOpenAI):
        completion_kwargs: dict[str, Any] = {"api_key": _secret_value(llm.openai_api_key)}
        if llm.openai_api_base:
            completion_kwargs["api_base"] = llm.openai_api_base
        if llm.default_headers:
            completion_kwargs["extra_headers"] = dict(llm.default_headers)
        if getattr(llm, "extra_body", None):
            completion_kwargs["extra_body"] = llm.extra_body
        return f"openai/{llm.model_name}", completion_kwargs

    raise ValueError(
        f"{type(llm).__name__} is not supported for DataRobot NAT evaluation judges. "
        "Use a workflow LLM that resolves to ChatLiteLLM, ChatOpenAI, or AzureChatOpenAI "
        "(for example ``datarobot-llm-component`` or DataRobot LLM Gateway)."
    )


def wrap_langchain_judge_for_llamaindex(llm: object) -> Any:
    """Wrap any LangChain chat model for LlamaIndex evaluators."""
    from datarobot_dome._import_utils import require_extra
    from langchain_core.language_models import BaseChatModel

    if not isinstance(llm, BaseChatModel):
        raise ValueError(
            f"{type(llm).__name__} is not a LangChain chat model and cannot be used "
            "for LlamaIndex-based evaluation."
        )
    try:
        from llama_index.core.llms import LLM
        from llama_index.llms.langchain import LangChainLLM
    except ImportError as e:
        raise require_extra("llama-index-llms-langchain", "llm-eval", e) from e

    return cast(LLM, LangChainLLM(llm=llm))
