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
from ag_ui.core import RunErrorEvent
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

from datarobot_genai.dragent.frontends.tool_call_registry import bind_tool_call
from datarobot_genai.dragent.frontends.tool_call_registry import defer_tool_end
from datarobot_genai.dragent.frontends.tool_call_registry import is_args_done
from datarobot_genai.dragent.frontends.tool_call_registry import pop_tool_call
from datarobot_genai.dragent.frontends.tool_call_registry import reset as reset_registry
from datarobot_genai.dragent.plugins.datarobot_dragent_normalization import (
    convert_chunks_to_agui_events,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_registry()
    yield
    reset_registry()


# --- Helpers ---


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
    events = []
    for resp in responses:
        events.extend(resp.events)
    return events


# --- Text message tests ---


class TestTextMessage:
    @pytest.mark.asyncio
    async def test_single_text_chunk_produces_start_content_end(self):
        chunk = _make_chunk(content="Hello")
        responses = await _collect(convert_chunks_to_agui_events(_async_iter(chunk)))
        events = _flat_events(responses)

        assert len(events) == 3
        assert isinstance(events[0], TextMessageStartEvent)
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
        responses = await _collect(convert_chunks_to_agui_events(_async_iter(*chunks)))
        events = _flat_events(responses)

        start_events = [e for e in events if isinstance(e, TextMessageStartEvent)]
        content_events = [e for e in events if isinstance(e, TextMessageContentEvent)]
        end_events = [e for e in events if isinstance(e, TextMessageEndEvent)]

        assert len(start_events) == 1
        assert len(content_events) == 2
        assert content_events[0].delta == "Hello "
        assert content_events[1].delta == "world"
        assert len(end_events) == 1


# --- Tool call tests ---


class TestToolCall:
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
        responses = await _collect(convert_chunks_to_agui_events(_async_iter(chunk)))
        events = _flat_events(responses)

        assert len(events) == 2
        assert isinstance(events[0], ToolCallStartEvent)
        assert events[0].tool_call_id == "tc-1"
        assert events[0].tool_call_name == "get_weather"
        assert isinstance(events[1], ToolCallArgsEvent)
        assert events[1].delta == '{"loc":'

    @pytest.mark.asyncio
    async def test_tool_call_followup_with_gemini_thought_id_keeps_first_id(self):
        """Gemini may append ``__thought__<sig>`` to tool ids on later chunks; first id wins."""
        base_id = "call_6464809420104580982e1b40c23b"
        thought_id = f"{base_id}__thought__AY89a1+sig"
        first = _make_chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id=base_id,
                    function=ChoiceDeltaToolCallFunction(name="planner", arguments=""),
                )
            ]
        )
        followup = _make_chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id=thought_id,
                    function=ChoiceDeltaToolCallFunction(arguments='{"topic":'),
                )
            ]
        )
        responses = await _collect(convert_chunks_to_agui_events(_async_iter(first, followup)))
        events = _flat_events(responses)

        starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        args_events = [e for e in events if isinstance(e, ToolCallArgsEvent)]
        assert len(starts) == 1
        assert starts[0].tool_call_id == base_id
        assert len(args_events) == 1
        assert args_events[0].tool_call_id == base_id

    @pytest.mark.asyncio
    async def test_tool_call_followup_chunks_use_index_lookup(self):
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
        responses = await _collect(convert_chunks_to_agui_events(_async_iter(first, followup)))
        events = _flat_events(responses)

        args_events = [e for e in events if isinstance(e, ToolCallArgsEvent)]
        assert len(args_events) == 2
        assert args_events[0].tool_call_id == "tc-1"
        assert args_events[1].tool_call_id == "tc-1"

    @pytest.mark.asyncio
    async def test_tool_call_after_text_threads_under_text_message_id(self):
        text = _make_chunk(content="Let me check.", chunk_id="msg-1")
        tool_call = _make_chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="tc-1",
                    function=ChoiceDeltaToolCallFunction(name="search", arguments="{}"),
                )
            ]
        )
        responses = await _collect(convert_chunks_to_agui_events(_async_iter(text, tool_call)))
        events = _flat_events(responses)

        start = next(e for e in events if isinstance(e, ToolCallStartEvent))
        assert start.parent_message_id == "msg-1"

    @pytest.mark.asyncio
    async def test_tool_call_with_no_preceding_text_uses_empty_parent(self):
        # Falls back to "" (langgraph convention), not a phantom uuid.
        chunk = _make_chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="tc-1",
                    function=ChoiceDeltaToolCallFunction(name="search", arguments="{}"),
                )
            ]
        )
        responses = await _collect(convert_chunks_to_agui_events(_async_iter(chunk)))
        events = _flat_events(responses)

        start = next(e for e in events if isinstance(e, ToolCallStartEvent))
        assert start.parent_message_id == ""

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
        responses = await _collect(convert_chunks_to_agui_events(_async_iter(chunk)))
        events = _flat_events(responses)

        start_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
        end_events = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(start_events) == 2
        assert {e.tool_call_id for e in start_events} == {"tc-1", "tc-2"}
        assert end_events == []


# --- Edge cases ---


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self):
        responses = await _collect(convert_chunks_to_agui_events(_async_iter()))
        assert responses == []

    @pytest.mark.asyncio
    async def test_chunk_with_no_content_and_no_tool_calls_is_skipped(self):
        chunk = _make_chunk(content=None, tool_calls=None)
        responses = await _collect(convert_chunks_to_agui_events(_async_iter(chunk)))
        assert responses == []


# --- Error handling ---


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_end_events_and_run_error_emitted_on_upstream_exception(self):
        """Upstream errors are absorbed and surfaced as RunErrorEvent, not propagated."""

        async def _failing_gen():
            yield _make_chunk(content="Hello")
            raise RuntimeError("upstream error")

        responses = await _collect(convert_chunks_to_agui_events(_failing_gen()))

        events = _flat_events(responses)
        assert any(isinstance(e, TextMessageStartEvent) for e in events)
        assert any(isinstance(e, TextMessageEndEvent) for e in events)
        error_events = [e for e in events if isinstance(e, RunErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].message == "upstream error"
        assert error_events[0].code == "STREAM_ERROR"

    @pytest.mark.asyncio
    async def test_aclose_during_stream_does_not_raise(self):
        async def _slow_gen():
            yield _make_chunk(content="Hello")
            yield _make_chunk(content=" world")

        gen = convert_chunks_to_agui_events(_slow_gen())
        await gen.__anext__()
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_stream_ending_with_active_tool_calls_emits_no_end(self):
        # Adaptor owns End; a post-loop End here would duplicate it.
        chunk = _make_chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="tc-1",
                    function=ChoiceDeltaToolCallFunction(name="search", arguments="{}"),
                )
            ]
        )
        responses = await _collect(convert_chunks_to_agui_events(_async_iter(chunk)))
        events = _flat_events(responses)

        assert [e for e in events if isinstance(e, ToolCallEndEvent)] == []

    @pytest.mark.asyncio
    async def test_stream_error_after_tool_call_emits_no_tool_call_end(self):
        # RunErrorEvent is terminal for the run; client closes any in-flight call.
        async def _failing_gen():
            yield _make_chunk(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="tc-1",
                        function=ChoiceDeltaToolCallFunction(name="search", arguments="{}"),
                    )
                ]
            )
            raise RuntimeError("boom")

        responses = await _collect(convert_chunks_to_agui_events(_failing_gen()))
        events = _flat_events(responses)

        assert [e for e in events if isinstance(e, ToolCallEndEvent)] == []
        assert any(isinstance(e, RunErrorEvent) for e in events)


# --- Registry integration ---


class TestToolCallRegistry:
    @pytest.mark.asyncio
    async def test_tool_call_start_publishes_to_registry(self):
        chunk = _make_chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="tc-1",
                    function=ChoiceDeltaToolCallFunction(name="planner", arguments="{}"),
                )
            ]
        )
        await _collect(convert_chunks_to_agui_events(_async_iter(chunk)))

        assert bind_tool_call("planner", "nat-uuid-1") == "tc-1"
        assert pop_tool_call("nat-uuid-1") == "tc-1"
        assert bind_tool_call("planner", "nat-uuid-2") is None

    @pytest.mark.asyncio
    async def test_content_after_tool_call_marks_args_done(self):
        """When text content follows tool calls, args are marked done."""
        tool_chunk = _make_chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="tc-1",
                    function=ChoiceDeltaToolCallFunction(name="search", arguments='{"q":"hi"}'),
                )
            ]
        )
        text_chunk = _make_chunk(content="Here are the results.", chunk_id="msg-2")
        await _collect(convert_chunks_to_agui_events(_async_iter(tool_chunk, text_chunk)))

        assert is_args_done("tc-1")

    @pytest.mark.asyncio
    async def test_content_after_tool_call_flushes_deferred_end_events(self):
        """Deferred end/result events are flushed when content arrives."""
        tool_chunk = _make_chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="tc-1",
                    function=ChoiceDeltaToolCallFunction(name="search", arguments="{}"),
                )
            ]
        )
        # Simulate the step adaptor deferring end events while args stream
        sentinel_end = ToolCallEndEvent(tool_call_id="tc-1")
        defer_tool_end("tc-1", [sentinel_end])

        text_chunk = _make_chunk(content="Done.", chunk_id="msg-2")
        responses = await _collect(
            convert_chunks_to_agui_events(_async_iter(tool_chunk, text_chunk))
        )
        events = _flat_events(responses)

        # The deferred ToolCallEndEvent should appear before the text content.
        end_events = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].tool_call_id == "tc-1"

    @pytest.mark.asyncio
    async def test_stream_end_marks_remaining_tool_calls_args_done(self):
        """When the stream ends with active tool calls, they are marked args-done."""
        chunk = _make_chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="tc-1",
                    function=ChoiceDeltaToolCallFunction(name="search", arguments="{}"),
                )
            ]
        )
        await _collect(convert_chunks_to_agui_events(_async_iter(chunk)))

        assert is_args_done("tc-1")
