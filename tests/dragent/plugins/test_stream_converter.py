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

import datetime
from collections.abc import AsyncGenerator

import pytest
from ag_ui.core import EventType
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallStartEvent
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import ChatResponseChunkChoice
from nat.data_models.api_server import ChoiceDelta
from nat.data_models.api_server import ChoiceDeltaToolCall
from nat.data_models.api_server import ChoiceDeltaToolCallFunction

from datarobot_genai.dragent.plugins.stream_converter import convert_chunks_to_ag_ui_events


def _make_chunk(
    content: str | None = None,
    tool_calls: list[ChoiceDeltaToolCall] | None = None,
    chunk_id: str = "chunk-1",
) -> ChatResponseChunk:
    return ChatResponseChunk(
        id=chunk_id,
        choices=[
            ChatResponseChunkChoice(
                index=0,
                delta=ChoiceDelta(content=content, tool_calls=tool_calls),
            )
        ],
        created=datetime.datetime.now(datetime.UTC),
    )


async def _async_iter(*items) -> AsyncGenerator[ChatResponseChunk, None]:
    for item in items:
        yield item


async def _collect(gen: AsyncGenerator) -> list:
    return [item async for item in gen]


def _flat_events(responses):
    """Flatten DRAgentEventResponse list into a flat list of AG-UI events."""
    events = []
    for resp in responses:
        events.extend(resp.events)
    return events


class TestTextMessageStream:
    @pytest.mark.asyncio
    async def test_single_text_chunk_produces_start_content_end(self):
        chunk = _make_chunk(content="Hello")
        responses = await _collect(convert_chunks_to_ag_ui_events(_async_iter(chunk)))
        events = _flat_events(responses)

        assert len(events) == 3
        assert isinstance(events[0], TextMessageStartEvent)
        assert events[0].type == EventType.TEXT_MESSAGE_START
        assert events[0].message_id == "chunk-1"
        assert isinstance(events[1], TextMessageContentEvent)
        assert events[1].delta == "Hello"
        assert isinstance(events[2], TextMessageEndEvent)

    @pytest.mark.asyncio
    async def test_multiple_text_chunks_share_single_start_end(self):
        chunks = [
            _make_chunk(content="Hello ", chunk_id="c1"),
            _make_chunk(content="world", chunk_id="c1"),
        ]
        responses = await _collect(convert_chunks_to_ag_ui_events(_async_iter(*chunks)))
        events = _flat_events(responses)

        start_events = [e for e in events if isinstance(e, TextMessageStartEvent)]
        content_events = [e for e in events if isinstance(e, TextMessageContentEvent)]
        end_events = [e for e in events if isinstance(e, TextMessageEndEvent)]

        assert len(start_events) == 1
        assert len(content_events) == 2
        assert content_events[0].delta == "Hello "
        assert content_events[1].delta == "world"
        assert len(end_events) == 1


class TestToolCallStream:
    @pytest.mark.asyncio
    async def test_single_tool_call_produces_start_args_end(self):
        chunk = _make_chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="tc-1",
                    function=ChoiceDeltaToolCallFunction(name="get_weather", arguments='{"loc":'),
                )
            ]
        )
        responses = await _collect(convert_chunks_to_ag_ui_events(_async_iter(chunk)))
        events = _flat_events(responses)

        assert len(events) == 3
        assert isinstance(events[0], ToolCallStartEvent)
        assert events[0].tool_call_id == "tc-1"
        assert events[0].tool_call_name == "get_weather"
        assert isinstance(events[1], ToolCallArgsEvent)
        assert events[1].delta == '{"loc":'
        assert isinstance(events[2], ToolCallEndEvent)

    @pytest.mark.asyncio
    async def test_tool_call_followup_chunks_use_index_lookup(self):
        """Follow-up chunks have id=None; the converter uses the index to look up the id."""
        first = _make_chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="tc-1",
                    function=ChoiceDeltaToolCallFunction(name="search", arguments='{"q":'),
                )
            ]
        )
        followup = _make_chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id=None,
                    function=ChoiceDeltaToolCallFunction(arguments='"hello"}'),
                )
            ]
        )
        responses = await _collect(convert_chunks_to_ag_ui_events(_async_iter(first, followup)))
        events = _flat_events(responses)

        start_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
        args_events = [e for e in events if isinstance(e, ToolCallArgsEvent)]
        end_events = [e for e in events if isinstance(e, ToolCallEndEvent)]

        assert len(start_events) == 1
        assert len(args_events) == 2
        assert args_events[0].tool_call_id == "tc-1"
        assert args_events[1].tool_call_id == "tc-1"
        assert args_events[1].delta == '"hello"}'
        assert len(end_events) == 1

    @pytest.mark.asyncio
    async def test_multiple_parallel_tool_calls(self):
        chunk = _make_chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="tc-1",
                    function=ChoiceDeltaToolCallFunction(name="tool_a", arguments="{}"),
                ),
                ChoiceDeltaToolCall(
                    index=1,
                    id="tc-2",
                    function=ChoiceDeltaToolCallFunction(name="tool_b", arguments="{}"),
                ),
            ]
        )
        responses = await _collect(convert_chunks_to_ag_ui_events(_async_iter(chunk)))
        events = _flat_events(responses)

        start_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
        end_events = [e for e in events if isinstance(e, ToolCallEndEvent)]

        assert len(start_events) == 2
        assert {e.tool_call_id for e in start_events} == {"tc-1", "tc-2"}
        assert len(end_events) == 2
        assert {e.tool_call_id for e in end_events} == {"tc-1", "tc-2"}


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self):
        responses = await _collect(convert_chunks_to_ag_ui_events(_async_iter()))
        assert responses == []

    @pytest.mark.asyncio
    async def test_non_chat_response_chunk_is_skipped(self):
        responses = await _collect(convert_chunks_to_ag_ui_events(_async_iter("not a chunk")))
        assert responses == []

    @pytest.mark.asyncio
    async def test_chunk_with_empty_choices_is_skipped(self):
        chunk = ChatResponseChunk(
            id="c1",
            choices=[],
            created=datetime.datetime.now(datetime.UTC),
        )
        responses = await _collect(convert_chunks_to_ag_ui_events(_async_iter(chunk)))
        assert responses == []

    @pytest.mark.asyncio
    async def test_chunk_with_no_content_and_no_tool_calls_is_skipped(self):
        chunk = _make_chunk(content=None, tool_calls=None)
        responses = await _collect(convert_chunks_to_ag_ui_events(_async_iter(chunk)))
        assert responses == []

    @pytest.mark.asyncio
    async def test_tool_call_without_function_gets_empty_name(self):
        chunk = _make_chunk(tool_calls=[ChoiceDeltaToolCall(index=0, id="tc-1", function=None)])
        responses = await _collect(convert_chunks_to_ag_ui_events(_async_iter(chunk)))
        events = _flat_events(responses)

        start_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
        assert len(start_events) == 1
        assert start_events[0].tool_call_name == ""
