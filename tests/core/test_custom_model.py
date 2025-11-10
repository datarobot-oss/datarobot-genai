# Copyright 2025 DataRobot, Inc. and its affiliates.
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

from collections.abc import AsyncGenerator
from collections.abc import Iterator
from typing import Any

from datarobot_genai.core.chat import CustomModelChatResponse
from datarobot_genai.core.chat import CustomModelStreamingResponse
from datarobot_genai.core.custom_model import chat_entrypoint
from datarobot_genai.core.custom_model import load_model


class _DummyAgent:
    def __init__(self, **_: Any) -> None:
        pass

    async def invoke(self, completion_create_params: Any) -> tuple[str, None, dict[str, int]]:
        assert "messages" in completion_create_params
        return ("ok", None, {})


class _DummyStreamingAgent:
    def __init__(self, **_: Any) -> None:
        pass

    async def invoke(
        self, completion_create_params: Any
    ) -> AsyncGenerator[tuple[str, None, dict[str, int]], None]:
        assert "messages" in completion_create_params

        async def _gen() -> AsyncGenerator[tuple[str, None, dict[str, int]], None]:
            yield ("a", None, {})
            yield ("b", None, {})
            return

        return _gen()


def _minimal_params() -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        # populate explicit empty auth context to avoid any external coupling
        "authorization_context": {},
    }


def test_chat_entrypoint_non_streaming() -> None:
    pool, loop = load_model()
    try:
        resp = chat_entrypoint(
            _DummyAgent,
            _minimal_params(),
            (pool, loop),
            work_dir=".",
            runtime_parameter_keys=[],
        )
        assert isinstance(resp, CustomModelChatResponse)
        assert resp.choices[0].message.content == "ok"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass


def test_chat_entrypoint_streaming() -> None:
    pool, loop = load_model()
    try:
        stream: Iterator[CustomModelStreamingResponse] | CustomModelChatResponse = chat_entrypoint(
            _DummyStreamingAgent,
            _minimal_params(),
            (pool, loop),
            work_dir=".",
            runtime_parameter_keys=[],
        )
        assert not isinstance(stream, CustomModelChatResponse)
        chunks = list(stream)  # type: ignore[arg-type]
        # Expect at least 2 deltas and one terminal chunk
        assert any(c.choices[0].delta.content == "a" for c in chunks if c.choices[0].delta)
        assert any(c.choices[0].delta.content == "b" for c in chunks if c.choices[0].delta)
        assert chunks[-1].choices[0].finish_reason == "stop"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
