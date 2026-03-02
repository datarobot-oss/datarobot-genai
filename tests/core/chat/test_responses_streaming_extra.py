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

from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ragas import MultiTurnSample
from ragas.messages import HumanMessage

from datarobot_genai.core.chat.responses import CustomModelStreamingResponse
from datarobot_genai.core.chat.responses import to_custom_model_streaming_response


async def _ok_stream() -> AsyncGenerator[tuple[Any, MultiTurnSample | None, dict[str, int]], None]:
    zero = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
    yield RunStartedEvent(thread_id="t", run_id="r"), None, zero
    yield TextMessageStartEvent(message_id="1", role="assistant"), None, zero
    yield (
        TextMessageContentEvent(message_id="1", delta="hello "),
        None,
        {
            "completion_tokens": 1,
            "prompt_tokens": 2,
            "total_tokens": 3,
        },
    )
    yield (
        TextMessageContentEvent(message_id="1", delta="world"),
        None,
        {
            "completion_tokens": 2,
            "prompt_tokens": 3,
            "total_tokens": 5,
        },
    )
    yield (
        TextMessageEndEvent(message_id="1"),
        None,
        {
            "completion_tokens": 2,
            "prompt_tokens": 3,
            "total_tokens": 5,
        },
    )
    yield (
        RunFinishedEvent(thread_id="t", run_id="r"),
        MultiTurnSample(user_input=[HumanMessage(content="x")]),
        {
            "completion_tokens": 2,
            "prompt_tokens": 3,
            "total_tokens": 5,
        },
    )


async def _err_stream() -> AsyncGenerator[tuple[Any, Any, dict[str, int]], None]:
    raise RuntimeError("boom")
    yield (
        RunStartedEvent(thread_id="t", run_id="r"),
        None,
        {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    )


def test_to_custom_model_streaming_response_success() -> None:
    loop = asyncio.new_event_loop()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            it = to_custom_model_streaming_response(pool, loop, _ok_stream(), model="m")
            chunks = list(it)
    finally:
        loop.close()

    # RunStarted/RunFinished are skipped; 4 event chunks + 1 final stop chunk = 5
    assert all(isinstance(c, CustomModelStreamingResponse) for c in chunks)
    content_chunks = [c for c in chunks if c.choices[0].delta.content]
    assert content_chunks[0].choices[0].delta.content == "hello "
    assert content_chunks[1].choices[0].delta.content == "world"
    # Final chunk has no content and includes pipeline_interactions
    assert chunks[-1].choices[0].delta.content is None
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].pipeline_interactions is not None


def test_to_custom_model_streaming_response_error() -> None:
    loop = asyncio.new_event_loop()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            it = to_custom_model_streaming_response(pool, loop, _err_stream(), model="m")
            chunks = list(it)
    finally:
        loop.close()

    # Single error chunk
    assert len(chunks) == 1
    assert chunks[0].choices[0].finish_reason == "stop"
    assert "boom" in (chunks[0].choices[0].delta.content or "")
