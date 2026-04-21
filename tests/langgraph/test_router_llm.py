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

"""Tests for RouterChatModel and get_router_llm (LangGraph)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

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


def _make_chunk(content: str) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


def _make_router(chunks: list[str]) -> MagicMock:
    router = MagicMock()
    router.completion.return_value = [_make_chunk(c) for c in chunks]
    router.acompletion = AsyncMock(return_value=_make_async_iter(chunks))
    return router


async def _make_async_iter(chunks: list[str]) -> Any:
    for c in chunks:
        yield _make_chunk(c)


# --- RouterChatModel ---


def test_router_chat_model_stream_yields_chunks() -> None:
    from datarobot_genai.langgraph.router_llm import RouterChatModel

    router = MagicMock()
    router.completion.return_value = [_make_chunk("Hello"), _make_chunk(" world")]
    model = RouterChatModel(router=router)

    chunks = list(model._stream([HumanMessage(content="hi")]))
    assert len(chunks) == 2
    assert chunks[0].message.content == "Hello"
    assert chunks[1].message.content == " world"


@pytest.mark.asyncio
async def test_router_chat_model_astream_yields_chunks() -> None:
    from datarobot_genai.langgraph.router_llm import RouterChatModel

    async def fake_acompletion(*args: Any, **kwargs: Any) -> Any:
        async def gen() -> Any:
            for c in ["Hello", " world"]:
                yield _make_chunk(c)

        return gen()

    router = MagicMock()
    router.acompletion = fake_acompletion
    model = RouterChatModel(router=router)

    chunks = [c async for c in model._astream([HumanMessage(content="hi")])]
    assert len(chunks) == 2
    contents = [c.message.content for c in chunks]
    assert "Hello" in contents
    assert " world" in contents


# --- get_router_llm ---


def test_get_router_llm_returns_base_chat_model() -> None:
    from datarobot_genai.langgraph.llm import get_router_llm

    primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
    _fallback = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")

    with patch("litellm.Router") as mock_router_cls:
        mock_router_cls.return_value = MagicMock()
        llm = get_router_llm(primary, [_fallback])

    assert isinstance(llm, BaseChatModel)
    assert llm._llm_type == "datarobot-router"
