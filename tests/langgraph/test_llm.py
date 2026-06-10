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

"""Tests for :mod:`datarobot_genai.langgraph.llm`."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel

from datarobot_genai.core.config import DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM
from datarobot_genai.core.config import LLMType
from datarobot_genai.langgraph import llm as langgraph_llm

pytestmark = pytest.mark.filterwarnings(
    "ignore:WARNING! api_base is not default parameter:UserWarning",
    "ignore:WARNING! stream_options is not default parameter:UserWarning",
)


@pytest.fixture(autouse=True)
def patched_langgraph_llm_defaults() -> None:
    with (
        patch.object(langgraph_llm, "default_api_key", return_value="sk-test-key"),
        patch.object(langgraph_llm, "default_model_name", return_value="default-model"),
        patch.object(
            langgraph_llm,
            "default_datarobot_llm_gateway_url",
            return_value="https://example.test/genai/llmgw",
        ),
        patch.object(
            langgraph_llm,
            "default_deployment_url",
            side_effect=lambda deployment_id: (
                f"https://example.test/deployments/{deployment_id}/chat/completions"
            ),
        ),
    ):
        yield


def test_module_exports_llm_factory_functions() -> None:
    assert callable(langgraph_llm.get_datarobot_gateway_llm)
    assert callable(langgraph_llm.get_datarobot_deployment_llm)
    assert callable(langgraph_llm.get_datarobot_nim_llm)


def test_get_datarobot_gateway_llm_returns_chat_openai_subclass() -> None:
    llm = langgraph_llm.get_datarobot_gateway_llm()
    assert isinstance(llm, BaseChatModel)
    assert llm.model == "datarobot/default-model"
    assert llm.api_base == "https://example.test/genai/llmgw"
    assert llm.stream_options == {"include_usage": True}


def test_get_datarobot_gateway_llm_strips_datarobot_model_prefix() -> None:
    llm = langgraph_llm.get_datarobot_gateway_llm("datarobot/azure/gpt-4")
    assert llm.model == "datarobot/azure/gpt-4"


def test_get_datarobot_gateway_llm_merges_parameters() -> None:
    llm = langgraph_llm.get_datarobot_gateway_llm(parameters={"temperature": 0.25})
    assert llm.temperature == 0.25


def test_get_datarobot_deployment_llm_sets_deployment_api_base() -> None:
    llm = langgraph_llm.get_datarobot_deployment_llm("dep-abc-123")
    assert isinstance(llm, BaseChatModel)
    assert llm.api_base == "https://example.test/deployments/dep-abc-123/chat/completions"
    assert llm.stream_options == {"include_usage": True}


def test_get_datarobot_deployment_llm_uses_deployed_placeholder_when_default_model_unset() -> None:
    with patch.object(langgraph_llm, "default_model_name", return_value=None):
        llm = langgraph_llm.get_datarobot_deployment_llm("dep-abc-123")
    assert llm.model == DEFAULT_MODEL_NAME_FOR_DEPLOYED_LLM


def test_get_datarobot_gateway_llm_raises_when_model_unset() -> None:
    with patch.object(langgraph_llm, "default_model_name", return_value=None):
        with pytest.raises(ValueError, match="Model name is required"):
            langgraph_llm.get_datarobot_gateway_llm()


def test_get_datarobot_nim_llm_builds_nim_endpoint_and_model() -> None:
    llm = langgraph_llm.get_datarobot_nim_llm("nim-1", "m", {"max_tokens": 10})
    assert isinstance(llm, BaseChatModel)
    assert llm.api_base == "https://example.test/deployments/nim-1/chat/completions"
    assert llm.model == "datarobot/m"


def test_get_datarobot_nim_llm_raises_when_model_unset() -> None:
    with patch.object(langgraph_llm, "default_model_name", return_value=None):
        with pytest.raises(ValueError, match="Model name is required"):
            langgraph_llm.get_datarobot_nim_llm("nim-1")


def test_get_external_llm_raises_when_model_unset() -> None:
    with patch.object(langgraph_llm, "default_model_name", return_value=None):
        with pytest.raises(ValueError, match="Model name is required"):
            langgraph_llm.get_external_llm()


def test_gateway_llm_factory_omits_stream_options_kwarg_when_not_streaming() -> None:
    llm = langgraph_llm.get_datarobot_gateway_llm(streaming=False)
    assert llm.streaming is False
    assert llm.stream_options is None


def test_gateway_llm_factory_passes_stream_options_when_streaming() -> None:
    llm = langgraph_llm.get_datarobot_gateway_llm(streaming=True)
    assert llm.streaming is True
    assert llm.stream_options == {"include_usage": True}


def test_get_external_llm_returns_base_chat_model() -> None:
    llm = langgraph_llm.get_external_llm()
    assert isinstance(llm, BaseChatModel)
    assert llm.model == "default-model"


def test_get_external_llm_strips_datarobot_prefix() -> None:
    llm = langgraph_llm.get_external_llm("datarobot/azure/gpt-4")
    assert llm.model == "azure/gpt-4"


def test_get_external_llm_preserves_model_without_prefix() -> None:
    llm = langgraph_llm.get_external_llm("azure/gpt-4")
    assert llm.model == "azure/gpt-4"


def test_get_external_llm_merges_parameters() -> None:
    llm = langgraph_llm.get_external_llm(parameters={"temperature": 0.7})
    assert llm.temperature == 0.7


def test_get_llm_routes_to_gateway() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.GATEWAY
    with patch.object(langgraph_llm, "Config", return_value=config):
        llm = langgraph_llm.get_llm()
    assert isinstance(llm, BaseChatModel)
    assert llm.api_base == "https://example.test/genai/llmgw"


def test_get_llm_routes_to_deployment() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.DEPLOYMENT
    config.llm_deployment_id = "dep-123"
    with patch.object(langgraph_llm, "Config", return_value=config):
        llm = langgraph_llm.get_llm()
    assert isinstance(llm, BaseChatModel)
    assert llm.api_base == "https://example.test/deployments/dep-123/chat/completions"


def test_get_llm_routes_to_nim() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.NIM
    config.nim_deployment_id = "nim-456"
    with patch.object(langgraph_llm, "Config", return_value=config):
        llm = langgraph_llm.get_llm()
    assert isinstance(llm, BaseChatModel)
    assert llm.api_base == "https://example.test/deployments/nim-456/chat/completions"


def test_get_llm_routes_to_external() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.EXTERNAL
    with patch.object(langgraph_llm, "Config", return_value=config):
        llm = langgraph_llm.get_llm()
    assert isinstance(llm, BaseChatModel)
    assert llm.model == "default-model"


def test_get_llm_raises_on_unknown_type() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = "unknown"
    with patch.object(langgraph_llm, "Config", return_value=config):
        with pytest.raises(ValueError, match="Invalid LLM type"):
            langgraph_llm.get_llm()


def test_get_llm_forwards_model_name_and_parameters() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.GATEWAY
    with patch.object(langgraph_llm, "Config", return_value=config):
        llm = langgraph_llm.get_llm(model_name="azure/gpt-4", parameters={"temperature": 0.5})
    assert llm.model == "datarobot/azure/gpt-4"
    assert llm.temperature == 0.5


def test_gateway_llm_forwards_extra_body_via_model_kwargs() -> None:
    llm = langgraph_llm.get_datarobot_gateway_llm(
        parameters={"extra_body": {"mock_response": "hello"}}
    )
    assert llm.model_kwargs.get("extra_body") == {"mock_response": "hello"}


def test_deployment_llm_forwards_extra_body_via_model_kwargs() -> None:
    llm = langgraph_llm.get_datarobot_deployment_llm(
        "dep-1", parameters={"extra_body": {"mock_response": "hello"}}
    )
    assert llm.model_kwargs.get("extra_body") == {"mock_response": "hello"}


def test_external_llm_forwards_extra_body_via_model_kwargs() -> None:
    llm = langgraph_llm.get_external_llm(parameters={"extra_body": {"mock_response": "hello"}})
    assert llm.model_kwargs.get("extra_body") == {"mock_response": "hello"}


def test_factory_preserves_existing_model_kwargs_when_adding_extra_body() -> None:
    llm = langgraph_llm.get_datarobot_gateway_llm(
        parameters={"extra_body": {"mock_response": "hello"}, "model_kwargs": {"top_k": 5}}
    )
    assert llm.model_kwargs == {"top_k": 5, "extra_body": {"mock_response": "hello"}}


def test_factory_omits_model_kwargs_when_no_extra_body() -> None:
    llm = langgraph_llm.get_datarobot_gateway_llm()
    assert llm.model_kwargs == {}


def test_get_llm_forwards_streaming_flag() -> None:
    config = MagicMock()
    config.get_llm_type.return_value = LLMType.GATEWAY
    with patch.object(langgraph_llm, "Config", return_value=config):
        llm = langgraph_llm.get_llm(streaming=False)
    assert llm.streaming is False


def test_wrap_bare_text_blocks_wraps_strings_and_keeps_dicts() -> None:
    """A bare string in list content (the answer langchain_litellm leaves behind after
    dropping thinking blocks) is wrapped as a text block so litellm's request transform
    never sees a non-dict item; existing dict blocks and plain-string content are kept.
    """
    message_dicts = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ["answer", {"type": "text", "text": "more"}]},
    ]
    out = langgraph_llm._wrap_bare_text_blocks(message_dicts)

    assert out[0]["content"] == "hi"  # scalar string content untouched
    assert out[1]["content"] == [
        {"type": "text", "text": "answer"},
        {"type": "text", "text": "more"},
    ]


def test_wrap_bare_text_blocks_preserves_multimodal_and_drops_empty() -> None:
    """Multimodal dict parts (e.g. ``image_url``) pass through untouched; a bare text
    string mixed in is wrapped as a text block; empty-string fragments are dropped.
    Wrapping (not gating on reasoning) means multimodal turns are never flattened.
    """
    message_dicts = [
        {
            "role": "user",
            "content": [
                "",
                "what is in this image?",
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KG"}},
            ],
        }
    ]
    out = langgraph_llm._wrap_bare_text_blocks(message_dicts)

    assert out[0]["content"] == [
        {"type": "text", "text": "what is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KG"}},
    ]


def test_create_message_dicts_wraps_reasoning_answer_as_text_block() -> None:
    """All four generation paths (sync/async, stream/generate) build their wire
    request through ``_create_message_dicts``, so overriding that one chokepoint
    covers every path. langchain_litellm drops the thinking blocks; the leftover
    answer is wrapped as a valid text block. The caller's history (and thus the
    response held in graph state) is left intact, so reasoning still renders.
    """
    from langchain_core.messages import AIMessage

    history = [
        AIMessage(
            content=[
                {"type": "thinking", "thinking": "prev"},
                {"type": "text", "text": "prior"},
            ]
        )
    ]
    llm = langgraph_llm.get_datarobot_deployment_llm("dep-1", model_name="m")
    message_dicts, _ = llm._create_message_dicts(history, None)

    # outgoing wire content is a valid text-block list (no bare string for litellm to choke on)
    assert message_dicts[0]["content"] == [{"type": "text", "text": "prior"}]
    # caller's history object not mutated
    assert isinstance(history[0].content, list)


def test_create_message_dicts_reasoning_only_message_is_empty() -> None:
    """A reasoning-only turn (the real gpt-oss/NIM shape: ``['', {thinking}...]``
    with no text block) serializes to an empty string on the wire instead of being
    re-sent as thinking blocks, which OpenAI-compatible backends reject.
    """
    from langchain_core.messages import AIMessage

    history = [
        AIMessage(
            content=[
                "",
                {"type": "thinking", "thinking": "Need"},
                {"type": "thinking", "thinking": " to search"},
            ]
        )
    ]
    llm = langgraph_llm.get_datarobot_deployment_llm("dep-1", model_name="m")
    message_dicts, _ = llm._create_message_dicts(history, None)

    assert message_dicts[0]["content"] == ""
