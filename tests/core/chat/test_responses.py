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
import asyncio
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from datarobot_genai.core.chat.responses import CustomModelChatResponse
from datarobot_genai.core.chat.responses import CustomModelStreamingResponse
from datarobot_genai.core.chat.responses import to_custom_model_chat_response
from datarobot_genai.core.chat.responses import to_custom_model_streaming_response


def test_to_custom_model_chat_response_basic() -> None:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    resp = to_custom_model_chat_response(
        response_text="Hello",
        pipeline_interactions=None,
        usage_metrics=usage,
        model="test-model",
    )
    assert isinstance(resp, CustomModelChatResponse)
    assert resp.choices[0].message.content == "Hello"
    assert resp.usage.prompt_tokens == 0
    assert resp.usage.completion_tokens == 0
    assert resp.usage.total_tokens == 0


def test_to_custom_model_streaming_response_sequence() -> None:
    async def gen() -> AsyncGenerator[tuple[str, Any | None, dict[str, int]], None]:
        yield ("Hello ", None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        yield ("World", None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        # final: no text, but returns last usage + last pipeline interactions when present
        yield (
            "",
            type("X", (), {"model_dump_json": lambda self: "{}"})(),
            {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )

    with ThreadPoolExecutor(1) as thread_pool_executor:
        event_loop = asyncio.new_event_loop()
        thread_pool_executor.submit(asyncio.set_event_loop, event_loop).result()

        response_generator = to_custom_model_streaming_response(
            thread_pool_executor, event_loop, gen(), model="m"
        )
        chunks = list(response_generator)
    assert isinstance(chunks[0], CustomModelStreamingResponse)
    assert chunks[0].choices[0].delta.content == "Hello "
    assert chunks[0].choices[0].finish_reason is None
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 3
