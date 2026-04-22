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

"""Tests for get_router_llm (LangGraph)."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk

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


def test_get_router_llm_returns_base_chat_model() -> None:
    from datarobot_genai.langgraph.llm import get_router_llm

    primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
    _fallback = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")

    with patch("litellm.Router") as mock_router_cls:
        mock_router_cls.return_value = MagicMock(model_list=[{"model_name": "primary"}])
        llm = get_router_llm(primary, [_fallback])

    assert isinstance(llm, BaseChatModel)


def test_get_router_llm_supports_bind_tools() -> None:
    from langchain_core.tools import tool

    from datarobot_genai.langgraph.llm import get_router_llm

    primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
    _fallback = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")

    @tool
    def dummy_tool(x: str) -> str:
        """Return x as-is."""
        return x

    with patch("litellm.Router") as mock_router_cls:
        mock_router_cls.return_value = MagicMock(model_list=[{"model_name": "primary"}])
        llm = get_router_llm(primary, [_fallback])

    bound = llm.bind_tools([dummy_tool])
    assert bound is not None


@pytest.mark.asyncio
async def test_get_router_llm_strips_tool_calls_from_additional_kwargs() -> None:
    """Streaming chunks must not carry raw tool-call deltas in additional_kwargs."""
    from datarobot_genai.langgraph.llm import get_router_llm

    primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
    fallback = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")

    fake_chunk = {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "my_tool", "arguments": '{"x":'},
                        }
                    ],
                }
            }
        ],
    }

    async def _fake_acompletion(**kwargs):  # noqa: ARG001
        yield fake_chunk

    with patch("litellm.Router") as mock_router_cls:
        mock_router = MagicMock(model_list=[{"model_name": "primary"}])
        mock_router.acompletion = AsyncMock(side_effect=_fake_acompletion)
        mock_router_cls.return_value = mock_router
        llm = get_router_llm(primary, [fallback])

    mock_router.acompletion = AsyncMock(side_effect=_fake_acompletion)

    chunks: list[AIMessageChunk] = []
    async for event in llm._astream([]):
        chunks.append(event.message)

    assert len(chunks) == 1
    assert "tool_calls" not in chunks[0].additional_kwargs
    assert len(chunks[0].tool_call_chunks) == 1
    assert chunks[0].tool_call_chunks[0]["name"] == "my_tool"
