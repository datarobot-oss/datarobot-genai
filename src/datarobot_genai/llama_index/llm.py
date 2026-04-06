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
from typing import Any
from datarobot_genai.core.config import Config
from datarobot_genai.core.utils.llm import _patch_llm_based_on_config


if TYPE_CHECKING:
    from llama_index.llms.litellm import LiteLLM


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


async def get_gateway_llm(llm_name: str) -> LiteLLM:
    llm_config = Config(llm_default_model=llm_name, streamming=True, use_datarobot_llm_gateway=True)
    config = {
        "base_url": llm_config.default_base_url(),
        "stream_options": {"include_usage": True},
        "api_key": llm_config.default_api_key(),
    }
    if not llm_config.llm_default_model.startswith("datarobot/"):
        config["model"] = "datarobot/" + llm_config.llm_default_model
    config["api_base"] = config.pop("base_url").removesuffix("/api/v2")
    client = _create_datarobot_litellm(config)
    yield _patch_llm_based_on_config(client, llm_config)


async def get_deployment_llm(deployment_id: str) -> LiteLLM:
    llm_config = Config(llm_deployment_id=deployment_id, streamming=True)
    config = {
        "base_url": llm_config.default_base_url(),
        "stream_options": {"include_usage": True},
        "api_key": llm_config.default_api_key()
    }
    if not llm_config.llm_default_model.startswith("datarobot/"):
        config["model"] = "datarobot/" + llm_config.llm_default_model
    config["api_base"] = config.pop("base_url") + "/chat/completions"

    # TODO do we need headers?
    if llm_config.headers:
        config["additional_kwargs"] = {"extra_headers": llm_config.headers}

    client = _create_datarobot_litellm(config)
    yield _patch_llm_based_on_config(client, llm_config)
