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
"""End-to-end: an agent that fails mid-stream surfaces as a framed terminal error on the wire.

Exercises the no-guards chain a failing dragent-native agent hits on the streaming routes:
``datarobot_otel_conventions`` (always-on, catches the exception -> terminal RUN_ERROR) ->
the route serializer. The guards-on chain (moderation wrapping otel) is covered by
``test_datarobot_moderation_middleware`` (stacked real-chain case). ``/generate/stream`` uses
NAT's interactive runner (a ``data:`` AG-UI RUN_ERROR); ``/chat/completions`` converts each item
to an OpenAI chunk via the registered converter. Nothing here is mocked except the runner's
execution-store/session plumbing.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageStartEvent
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import GlobalTypeConverter
from nat.front_ends.fastapi.http_interactive_runner import HTTPInteractiveRunner

# Importing register applies the DRAgentEventResponse -> ChatResponseChunk converter registration.
import datarobot_genai.dragent.frontends.register  # noqa: E402, F401
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.plugins.datarobot_otel_conventions_middleware import (
    DataRobotOtelConventionsMiddleware,
)


async def _failing_agent(*args: Any, **kwargs: Any) -> AsyncIterator[DRAgentEventResponse]:
    """dragent-native agent: streams a partial answer, then fails mid-stream."""
    yield DRAgentEventResponse(events=[TextMessageStartEvent(message_id="m1")])
    yield DRAgentEventResponse(
        events=[TextMessageContentEvent(message_id="m1", delta="partial answer")]
    )
    raise RuntimeError("tool timed out")


def _otel_wrapped_failing_stream() -> AsyncIterator[DRAgentEventResponse]:
    """Return the failing agent wrapped by the always-on otel middleware."""
    otel = DataRobotOtelConventionsMiddleware(MagicMock(), MagicMock())
    return otel.function_middleware_stream(
        MagicMock(), call_next=_failing_agent, context=MagicMock()
    )


def _interactive_runner() -> HTTPInteractiveRunner:
    """Build a real HTTPInteractiveRunner with only execution-store/session plumbing mocked."""
    store = MagicMock()
    store.create_execution = AsyncMock(return_value=MagicMock())

    @asynccontextmanager
    async def _fake_session(**kwargs: Any) -> AsyncIterator[Any]:
        yield MagicMock()

    session_manager = MagicMock()
    session_manager.session = _fake_session
    return HTTPInteractiveRunner(
        execution_store=store, session_manager=session_manager, http_flow_handler=None
    )


async def test_generate_stream_failure_is_agui_run_error_data_frame() -> None:
    """/generate/stream (interactive): a mid-stream failure is a ``data:`` AG-UI RUN_ERROR.

    Not NAT's ``event: error`` frame, which AG-UI clients don't recognize as a run failure.
    """
    runner = _interactive_runner()
    frames = [
        frame
        async for frame in runner._streaming_generator_impl(
            MagicMock(),
            workflow_gen_factory=lambda _session: _otel_wrapped_failing_stream(),
            error_log_message="test",
            passthrough_str_items=False,
        )
    ]

    assert frames, "expected streamed frames"
    assert not any("event: error" in frame for frame in frames), frames
    terminal = frames[-1]
    assert terminal.startswith("data:")
    payload = json.loads(terminal[len("data: ") :])
    error_event = payload["events"][0]
    assert error_event["type"] == "RUN_ERROR"
    assert error_event["message"] == "tool timed out"
    assert error_event["code"] == "RUN_ERROR"


async def test_chat_completions_failure_is_openai_error_chunk() -> None:
    """/chat/completions: the same failure adapts to an OpenAI-shaped error chunk.

    Mirrors NAT's chat serializer (``result_stream(to_type=ChatResponseChunk)`` -> converter).
    """
    responses = [response async for response in _otel_wrapped_failing_stream()]
    frames = [
        GlobalTypeConverter.convert(response, to_type=ChatResponseChunk).get_stream_data()
        for response in responses
    ]

    assert frames, "expected converted frames"
    error = json.loads(frames[-1][len("data: ") :])
    assert error["object"] == "chat.completion.chunk"
    assert error["choices"] == []
    assert error["error"] == {
        "message": "tool timed out",
        "type": "workflow_error",
        "code": "RUN_ERROR",
    }
    # OpenAI-adapted frame, not a raw AG-UI RUN_ERROR event (which would carry an ``events`` list).
    assert "events" not in error


async def test_chat_completions_non_streaming_failure_raises_through_type_converter() -> None:
    """Non-streaming /chat/completions: the aggregated RUN_ERROR raises through NAT's
    ``GlobalTypeConverter`` (``to_type=ChatResponse``), which the route returns as HTTP 422,
    not a swallowed empty-success.
    """
    responses = [response async for response in _otel_wrapped_failing_stream()]
    aggregated = DRAgentEventResponse(
        events=[event for response in responses for event in response.events],
    )

    with pytest.raises(RuntimeError, match="tool timed out"):
        GlobalTypeConverter.convert(aggregated, to_type=ChatResponse)
