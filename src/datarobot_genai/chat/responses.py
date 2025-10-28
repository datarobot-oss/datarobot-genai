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

import time
import uuid
from collections.abc import Generator
from collections.abc import Iterator

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from ragas import MultiTurnSample


class CustomModelChatResponse(ChatCompletion):
    pipeline_interactions: str | None = None


class CustomModelStreamingResponse(ChatCompletionChunk):
    pipeline_interactions: str | None = None


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
    streaming_response_generator: Generator[
        tuple[str, MultiTurnSample | None, dict[str, int]], None, None
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

    required_usage_metrics: dict[str, int] = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
    }

    for (
        response_text,
        pipeline_interactions,
        usage_metrics,
    ) in streaming_response_generator:
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
                usage=CompletionUsage.model_validate(required_usage_metrics | usage_metrics)
                if usage_metrics
                else None,
            )

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
