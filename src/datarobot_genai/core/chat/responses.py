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

"""OpenAI-compatible response helpers for chat interactions."""

from __future__ import annotations

import asyncio
import queue
import time
import traceback as tb
import uuid
from asyncio import AbstractEventLoop
from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

from ag_ui.core import BaseEvent
from ag_ui.core import Event
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import TextMessageContentEvent
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from datarobot_genai.core.agents import default_usage_metrics

if TYPE_CHECKING:
    from ragas import MultiTurnSample


class CustomModelChatResponse(ChatCompletion):
    pipeline_interactions: str | None = None


class CustomModelStreamingResponse(ChatCompletionChunk):
    pipeline_interactions: str | None = None
    event: Event | None = None


def to_custom_model_chat_response(
    response_text: str,
    pipeline_interactions: MultiTurnSample | None,
    usage_metrics: dict[str, int],
    model: str | object | None,
) -> CustomModelChatResponse:
    """Convert the OpenAI ChatCompletion response to CustomModelChatResponse."""
    choice = Choice(
        index=0,
        message=ChatCompletionMessage(role="assistant", content=response_text),
        finish_reason="stop",
    )

    if model is None:
        model = "unspecified-model"
    else:
        model = str(model)

    required_usage_metrics: dict[str, int] = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
    }

    return CustomModelChatResponse(
        id=str(uuid.uuid4()),
        object="chat.completion",
        choices=[choice],
        created=int(time.time()),
        model=model,
        usage=CompletionUsage.model_validate(required_usage_metrics | usage_metrics),
        pipeline_interactions=pipeline_interactions.model_dump_json()
        if pipeline_interactions
        else None,
    )


def to_custom_model_streaming_response(
    thread_pool_executor: ThreadPoolExecutor,
    event_loop: AbstractEventLoop,
    streaming_response_generator: AsyncGenerator[
        tuple[str | Event, MultiTurnSample | None, dict[str, int]], None
    ],
    model: str | object | None,
) -> Iterator[CustomModelStreamingResponse]:
    """Convert the OpenAI ChatCompletionChunk response to CustomModelStreamingResponse."""
    completion_id = str(uuid.uuid4())
    created = int(time.time())

    last_pipeline_interactions = None
    last_usage_metrics = None

    if model is None:
        model = "unspecified-model"
    else:
        model = str(model)

    required_usage_metrics = default_usage_metrics()
    try:
        agent_response = aiter(streaming_response_generator)
        while True:
            try:
                (
                    response_text_or_event,
                    pipeline_interactions,
                    usage_metrics,
                ) = thread_pool_executor.submit(
                    event_loop.run_until_complete, anext(agent_response)
                ).result()
                last_pipeline_interactions = pipeline_interactions
                last_usage_metrics = usage_metrics

                if isinstance(response_text_or_event, str) and response_text_or_event:
                    choice = ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(role="assistant", content=response_text_or_event),
                        finish_reason=None,
                    )
                    yield CustomModelStreamingResponse(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=model,
                        choices=[choice],
                        usage=CompletionUsage.model_validate(required_usage_metrics | usage_metrics)
                        if usage_metrics
                        else None,
                    )
                elif isinstance(response_text_or_event, BaseEvent):
                    content = ""
                    if isinstance(
                        response_text_or_event, (TextMessageContentEvent, TextMessageChunkEvent)
                    ):
                        content = response_text_or_event.delta or content
                    choice = ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(role="assistant", content=content),
                        finish_reason=None,
                    )

                    yield CustomModelStreamingResponse(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=model,
                        choices=[choice],
                        usage=CompletionUsage.model_validate(required_usage_metrics | usage_metrics)
                        if usage_metrics
                        else None,
                        event=response_text_or_event,
                    )
            except StopAsyncIteration:
                break
        event_loop.run_until_complete(streaming_response_generator.aclose())
        # Yield final chunk indicating end of stream
        choice = ChunkChoice(
            index=0,
            delta=ChoiceDelta(role="assistant"),
            finish_reason="stop",
        )
        yield CustomModelStreamingResponse(
            id=completion_id,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[choice],
            usage=CompletionUsage.model_validate(required_usage_metrics | last_usage_metrics)
            if last_usage_metrics
            else None,
            pipeline_interactions=last_pipeline_interactions.model_dump_json()
            if last_pipeline_interactions
            else None,
        )
    except Exception as e:
        tb.print_exc()
        created = int(time.time())
        choice = ChunkChoice(
            index=0,
            delta=ChoiceDelta(role="assistant", content=str(e), refusal="error"),
            finish_reason="stop",
        )
        yield CustomModelStreamingResponse(
            id=completion_id,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[choice],
            usage=None,
        )


def streaming_iterator_to_custom_model_streaming_response(
    streaming_response_iterator: Iterator[tuple[str, MultiTurnSample | None, dict[str, int]]],
    model: str | object | None,
) -> Iterator[CustomModelStreamingResponse]:
    """Convert the OpenAI ChatCompletionChunk response to CustomModelStreamingResponse."""
    completion_id = str(uuid.uuid4())
    created = int(time.time())

    last_pipeline_interactions = None
    last_usage_metrics = None

    while True:
        try:
            (
                response_text,
                pipeline_interactions,
                usage_metrics,
            ) = next(streaming_response_iterator)
            last_pipeline_interactions = pipeline_interactions
            last_usage_metrics = usage_metrics

            if response_text:
                choice = ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=response_text),
                    finish_reason=None,
                )
                yield CustomModelStreamingResponse(
                    id=completion_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=model,
                    choices=[choice],
                    usage=CompletionUsage(**usage_metrics) if usage_metrics else None,
                )
        except StopIteration:
            break
    # Yield final chunk indicating end of stream
    choice = ChunkChoice(
        index=0,
        delta=ChoiceDelta(role="assistant"),
        finish_reason="stop",
    )
    yield CustomModelStreamingResponse(
        id=completion_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[choice],
        usage=CompletionUsage(**last_usage_metrics) if last_usage_metrics else None,
        pipeline_interactions=last_pipeline_interactions.model_dump_json()
        if last_pipeline_interactions
        else None,
    )


T = TypeVar("T")


def async_gen_to_sync_thread(
    async_iterator: AsyncIterator[T],
    thread_pool_executor: ThreadPoolExecutor,
    event_loop: asyncio.AbstractEventLoop,
) -> Iterator[T]:
    """Run an async iterator in a separate thread and provide a sync iterator."""
    # A thread-safe queue for communication
    sync_queue: queue.Queue[Any] = queue.Queue()
    # A sentinel object to signal the end of the async generator
    SENTINEL = object()  # noqa: N806

    async def run_async_to_queue() -> None:
        """Run in the separate thread's event loop."""
        try:
            async for item in async_iterator:
                sync_queue.put(item)
        except Exception as e:
            # Put the exception on the queue to be re-raised in the main thread
            sync_queue.put(e)
        finally:
            # Signal the end of iteration
            sync_queue.put(SENTINEL)

    thread_pool_executor.submit(event_loop.run_until_complete, run_async_to_queue()).result()

    # The main thread consumes items synchronously
    while True:
        item = sync_queue.get()
        if item is SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item
