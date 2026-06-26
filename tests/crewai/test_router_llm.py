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

"""Tests for get_router_llm (CrewAI)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from crewai import LLM

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


def _make_tool_chunk() -> Any:
    function = SimpleNamespace(name="generate_objectid", arguments='{"type": "deployment"}')
    tool_call = SimpleNamespace(index=0, id="call_1", type="function", function=function)
    delta = SimpleNamespace(content=None, tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def test_get_router_llm_returns_llm_instance() -> None:
    from datarobot_genai.crewai.llm import get_router_llm

    primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
    _fallback = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")

    with patch("litellm.Router") as mock_router_cls:
        mock_router_cls.return_value = MagicMock()
        llm = get_router_llm(primary, [_fallback])

    assert isinstance(llm, LLM)
    assert llm.is_litellm is True


def test_router_llm_call_streams_and_accumulates() -> None:
    from datarobot_genai.crewai.llm import get_router_llm

    chunks = [_make_chunk("Hello"), _make_chunk(" world")]

    mock_router = MagicMock()
    mock_router.completion.return_value = iter(chunks)

    with patch("litellm.Router", return_value=mock_router):
        primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
        _fb = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")
        llm = get_router_llm(primary, [_fb])

    result = llm.call(messages=[{"role": "user", "content": "hi"}])
    assert result == "Hello world"
    # Verify streaming mode was used
    mock_router.completion.assert_called_once()
    assert mock_router.completion.call_args.kwargs.get("stream") is True


def test_router_llm_call_returns_tool_call_list_not_json_string() -> None:
    """When the model emits tool calls, ``call`` returns the bare list CrewAI's native loop
    executes — not a json string, which CrewAI would treat as a final answer (tool never runs).
    """
    from datarobot_genai.crewai.llm import get_router_llm

    mock_router = MagicMock()
    mock_router.completion.return_value = iter([_make_tool_chunk()])
    with patch("litellm.Router", return_value=mock_router):
        primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
        _fb = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")
        llm = get_router_llm(primary, [_fb])

    result = llm.call(messages=[{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert isinstance(result, list)
    assert result[0]["function"]["name"] == "generate_objectid"


async def test_router_llm_acall_returns_tool_call_list_not_json_string() -> None:
    """Async variant: ``acall`` returns the tool-call list, not a json string."""
    from datarobot_genai.crewai.llm import get_router_llm

    async def fake_acompletion(*args: Any, **kwargs: Any) -> Any:
        async def gen() -> Any:
            yield _make_tool_chunk()

        return gen()

    mock_router = MagicMock()
    mock_router.acompletion = fake_acompletion
    with patch("litellm.Router", return_value=mock_router):
        primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
        _fb = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")
        llm = get_router_llm(primary, [_fb])

    result = await llm.acall(
        messages=[{"role": "user", "content": "hi"}], tools=[{"type": "function"}]
    )
    assert isinstance(result, list)
    assert result[0]["function"]["name"] == "generate_objectid"


def test_router_llm_call_invokes_callbacks_per_chunk() -> None:
    from datarobot_genai.crewai.llm import get_router_llm

    chunks = [_make_chunk("A"), _make_chunk("B")]
    mock_router = MagicMock()
    mock_router.completion.return_value = iter(chunks)

    with patch("litellm.Router", return_value=mock_router):
        primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
        _fb = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")
        llm = get_router_llm(primary, [_fb])

    callback = MagicMock()
    callback.on_llm_new_token = MagicMock()
    llm.call(messages=[{"role": "user", "content": "hi"}], callbacks=[callback])
    assert callback.on_llm_new_token.call_count == 2


def test_router_llm_call_emits_llm_stream_chunk_events() -> None:
    """Verify call() emits LLMStreamChunkEvent so CrewAI's CrewStreamingOutput receives tokens."""
    from crewai.events import crewai_event_bus
    from crewai.events.types.llm_events import LLMStreamChunkEvent

    from datarobot_genai.crewai.llm import get_router_llm

    chunks = [_make_chunk("Hello"), _make_chunk(" world")]
    mock_router = MagicMock()
    mock_router.completion.return_value = iter(chunks)

    with patch("litellm.Router", return_value=mock_router):
        primary = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-1")
        _fb = LLMConfig(use_datarobot_llm_gateway=False, llm_deployment_id="dep-2")
        llm = get_router_llm(primary, [_fb])

    emitted: list[LLMStreamChunkEvent] = []

    def capture(_: object, event: object) -> None:
        if isinstance(event, LLMStreamChunkEvent):
            emitted.append(event)

    with crewai_event_bus.scoped_handlers():
        crewai_event_bus.register_handler(LLMStreamChunkEvent, capture)
        llm.call(messages=[{"role": "user", "content": "hi"}])

    assert len(emitted) == 2
    assert emitted[0].chunk == "Hello"
    assert emitted[1].chunk == " world"


@pytest.mark.asyncio
async def test_router_llm_acall_streams_and_accumulates() -> None:
    from datarobot_genai.crewai.llm import get_router_llm

    chunks = [_make_chunk("Async"), _make_chunk(" result")]

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

    result = await llm.acall(messages=[{"role": "user", "content": "hi"}])
    assert result == "Async result"


@pytest.mark.asyncio
async def test_router_llm_acall_emits_llm_stream_chunk_events() -> None:
    """Verify acall() emits LLMStreamChunkEvent for Crew.akickoff streaming."""
    from crewai.events import crewai_event_bus
    from crewai.events.types.llm_events import LLMStreamChunkEvent

    from datarobot_genai.crewai.llm import get_router_llm

    chunks = [_make_chunk("Hello"), _make_chunk(" world")]

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

    emitted: list[LLMStreamChunkEvent] = []

    def capture(_: object, event: object) -> None:
        if isinstance(event, LLMStreamChunkEvent):
            emitted.append(event)

    with crewai_event_bus.scoped_handlers():
        crewai_event_bus.register_handler(LLMStreamChunkEvent, capture)
        await llm.acall(messages=[{"role": "user", "content": "hi"}])

    assert len(emitted) == 2
    assert emitted[0].chunk == "Hello"
    assert emitted[1].chunk == " world"
