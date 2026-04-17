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

"""Tests for get_router_llm (LlamaIndex)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from llama_index.llms.litellm import LiteLLM

from datarobot_genai.core import config as config_mod
from datarobot_genai.core.config import LLMConfig


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch: pytest.MonkeyPatch) -> None:
    env = config_mod.Config.model_construct(
        datarobot_endpoint="https://app.datarobot.com/api/v2",
        datarobot_api_token="env-token",
        llm_deployment_id=None,
        nim_deployment_id=None,
        use_datarobot_llm_gateway=True,
        llm_default_model=None,
    )
    monkeypatch.setattr(config_mod, "Config", lambda: env)


def _make_resp_chunk(content: str) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


def test_get_router_llm_returns_litellm_instance() -> None:
    from datarobot_genai.llama_index.llm import get_router_llm

    primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
    fb = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")

    with patch("litellm.Router") as mock_cls:
        mock_cls.return_value = MagicMock()
        llm = get_router_llm(primary, [fb])

    assert isinstance(llm, LiteLLM)
    assert llm.metadata.is_function_calling_model is True
    assert llm.metadata.is_chat_model is True


def test_get_router_llm_stream_chat_yields_chunks() -> None:
    from datarobot_genai.llama_index.llm import get_router_llm

    chunks = [_make_resp_chunk("Hello"), _make_resp_chunk(" world")]
    mock_router = MagicMock()
    mock_router.completion.return_value = iter(chunks)

    with patch("litellm.Router", return_value=mock_router):
        primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
        fb = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")
        llm = get_router_llm(primary, [fb])

    from llama_index.core.base.llms.types import ChatMessage

    responses = list(llm._stream_chat([ChatMessage(role="user", content="hi")]))
    assert len(responses) == 2
    assert "Hello" in responses[0].delta or responses[0].delta == "Hello"
    assert mock_router.completion.call_args.kwargs.get("stream") is True


@pytest.mark.asyncio
async def test_get_router_llm_astream_chat_yields_chunks() -> None:
    from datarobot_genai.llama_index.llm import get_router_llm

    chunks = [_make_resp_chunk("Async"), _make_resp_chunk(" stream")]

    async def fake_acompletion(*args: Any, **kwargs: Any) -> Any:
        async def gen() -> Any:
            for c in chunks:
                yield c
        return gen()

    mock_router = MagicMock()
    mock_router.acompletion = fake_acompletion

    with patch("litellm.Router", return_value=mock_router):
        primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
        fb = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")
        llm = get_router_llm(primary, [fb])

    from llama_index.core.base.llms.types import ChatMessage

    gen = await llm._astream_chat([ChatMessage(role="user", content="hi")])
    results = [r async for r in gen]
    assert len(results) == 2
