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

from ragas import MultiTurnSample
from ragas.messages import HumanMessage

from datarobot_genai.core.chat.responses import CustomModelStreamingResponse
from datarobot_genai.core.chat.responses import to_custom_model_streaming_response


async def _ok_stream() -> AsyncGenerator[tuple[str, MultiTurnSample | None, dict[str, int]], None]:
    yield "hello ", None, {"completion_tokens": 1, "prompt_tokens": 2, "total_tokens": 3}
    yield (
        "world",
        MultiTurnSample(user_input=[HumanMessage(content="x")]),
        {
            "completion_tokens": 2,
            "prompt_tokens": 3,
            "total_tokens": 5,
        },
    )


async def _err_stream() -> AsyncGenerator[tuple[str, Any, dict[str, int]], None]:
    raise RuntimeError("boom")
    yield "", None, {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}


def test_to_custom_model_streaming_response_success() -> None:
    loop = asyncio.new_event_loop()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            it = to_custom_model_streaming_response(pool, loop, _ok_stream(), model="m")
            chunks = list(it)
    finally:
        loop.close()

    # Expect two chunks with content and one final empty chunk with pipeline_interactions
    assert len(chunks) == 3
    assert all(isinstance(c, CustomModelStreamingResponse) for c in chunks)
    assert chunks[0].choices[0].delta.content == "hello "
    assert chunks[1].choices[0].delta.content == "world"
    # Final chunk has no content and includes pipeline_interactions
    assert chunks[2].choices[0].delta.content is None
    assert chunks[2].choices[0].finish_reason == "stop"
    assert chunks[2].pipeline_interactions is not None


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
