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
import json
from collections.abc import AsyncIterator

import pytest
from ag_ui.core import Event
from ag_ui.core import EventType
from ag_ui.core import RunFinishedEvent
from ag_ui.core import RunStartedEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent

from datarobot_genai.core.agents.base import UsageMetrics
from datarobot_genai.core.agents.text_tool_calls import HermesTextToolCallParser
from datarobot_genai.core.agents.text_tool_calls import TextSegment
from datarobot_genai.core.agents.text_tool_calls import ToolCallSegment
from datarobot_genai.core.agents.text_tool_calls import ToolResultSegment
from datarobot_genai.core.agents.text_tool_calls import rewrite_text_tool_calls

USAGE: UsageMetrics = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

TOOL_CALL_MARKUP = (
    '<tool_call> {"name": "tavily_search", "arguments": {"query": "otel"}} </tool_call>'
)
TOOL_RESPONSE_MARKUP = '<tool_response> [{"url": "https://x"}] </tool_response>'


# --------------------------------------------------------------------------- #
# Pure parser
# --------------------------------------------------------------------------- #
def test_plain_text_passes_through() -> None:
    parser = HermesTextToolCallParser()
    segments = parser.feed("Just some prose.") + parser.flush()
    assert segments == [TextSegment("Just some prose.")]


def test_single_call_in_one_delta() -> None:
    parser = HermesTextToolCallParser()
    segments = parser.feed(TOOL_CALL_MARKUP) + parser.flush()
    assert len(segments) == 1
    call = segments[0]
    assert isinstance(call, ToolCallSegment)
    assert call.name == "tavily_search"
    assert json.loads(call.arguments) == {"query": "otel"}


def test_call_then_response_with_surrounding_prose() -> None:
    parser = HermesTextToolCallParser()
    text = f"Let me look. {TOOL_CALL_MARKUP} {TOOL_RESPONSE_MARKUP} Done."
    segments = parser.feed(text) + parser.flush()
    # The parser preserves inter-tag whitespace as text segments; dropping that
    # whitespace is the AG-UI layer's job (see the rewrite tests below).
    meaningful = [s for s in segments if not (isinstance(s, TextSegment) and not s.text.strip())]
    assert [type(s) for s in meaningful] == [
        TextSegment,
        ToolCallSegment,
        ToolResultSegment,
        TextSegment,
    ]
    assert meaningful[0] == TextSegment("Let me look. ")
    assert isinstance(meaningful[2], ToolResultSegment)
    assert json.loads(meaningful[2].content) == [{"url": "https://x"}]
    assert meaningful[3] == TextSegment(" Done.")


def test_markup_split_across_deltas() -> None:
    parser = HermesTextToolCallParser()
    segments: list = []
    # Split the opening tag, the body, and the closing tag across chunks.
    for chunk in ["pre <tool_", 'call> {"name": "s", "arg', 'uments": {}} </tool', "_call> post"]:
        segments += parser.feed(chunk)
    segments += parser.flush()
    kinds = [type(s) for s in segments]
    assert ToolCallSegment in kinds
    call = next(s for s in segments if isinstance(s, ToolCallSegment))
    assert call.name == "s"
    # No raw markup leaks into any text segment.
    text = "".join(s.text for s in segments if isinstance(s, TextSegment))
    assert "<tool_call>" not in text
    assert "</tool_call>" not in text
    assert "pre " in text
    assert " post" in text


def test_partial_opener_is_held_back_then_resolves() -> None:
    parser = HermesTextToolCallParser()
    # A trailing "<" must not be emitted as text — it could begin a tag.
    first = parser.feed("hello <")
    assert first == [TextSegment("hello ")]
    rest = parser.feed('tool_call> {"name": "t", "arguments": {}} </tool_call>')
    rest += parser.flush()
    assert any(isinstance(s, ToolCallSegment) for s in rest)


def test_invalid_json_call_falls_back_to_text() -> None:
    parser = HermesTextToolCallParser()
    segments = parser.feed("<tool_call> not json </tool_call>") + parser.flush()
    assert segments == [TextSegment("<tool_call> not json </tool_call>")]


def test_unterminated_block_flushed_as_text() -> None:
    parser = HermesTextToolCallParser()
    segments = parser.feed('text <tool_call> {"name": "t"') + parser.flush()
    text = "".join(s.text for s in segments if isinstance(s, TextSegment))
    assert text == 'text <tool_call> {"name": "t"'


def test_arguments_given_as_json_string_not_double_encoded() -> None:
    parser = HermesTextToolCallParser()
    markup = '<tool_call> {"name": "t", "arguments": "{\\"q\\": 1}"} </tool_call>'
    segments = parser.feed(markup) + parser.flush()
    call = segments[0]
    assert isinstance(call, ToolCallSegment)
    assert json.loads(call.arguments) == {"q": 1}


# --------------------------------------------------------------------------- #
# AG-UI rewrite transform (framework-agnostic, applied at the shared boundary)
# --------------------------------------------------------------------------- #
async def _stream(events: list[Event]) -> AsyncIterator[tuple[Event, None, UsageMetrics]]:
    for event in events:
        yield event, None, USAGE


async def _drain(events: list[Event]) -> list[Event]:
    return [item[0] async for item in rewrite_text_tool_calls(_stream(events))]


@pytest.mark.asyncio
async def test_rewrite_recovers_tool_events_from_text() -> None:
    upstream = [
        RunStartedEvent(type=EventType.RUN_STARTED, thread_id="t", run_id="r"),
        TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id="m1"),
        TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT, message_id="m1", delta="Searching. "
        ),
        TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT, message_id="m1", delta=TOOL_CALL_MARKUP
        ),
        TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT, message_id="m1", delta=TOOL_RESPONSE_MARKUP
        ),
        TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT, message_id="m1", delta="All done."
        ),
        TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id="m1"),
        RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id="t", run_id="r"),
    ]
    out = await _drain(upstream)
    types = [e.type for e in out]
    assert types == [
        EventType.RUN_STARTED,
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_ARGS,
        EventType.TOOL_CALL_END,
        EventType.TOOL_CALL_RESULT,
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
        EventType.RUN_FINISHED,
    ]

    start, args, end, result = out[4], out[5], out[6], out[7]
    assert start.tool_call_name == "tavily_search"
    assert start.tool_call_id == args.tool_call_id == end.tool_call_id == result.tool_call_id
    assert json.loads(args.delta) == {"query": "otel"}
    assert json.loads(result.content) == [{"url": "https://x"}]

    # No raw markup survives in any text content.
    text = "".join(e.delta for e in out if e.type == EventType.TEXT_MESSAGE_CONTENT)
    assert "<tool_call>" not in text and "<tool_response>" not in text
    assert "Searching." in text and "All done." in text


@pytest.mark.asyncio
async def test_rewrite_is_noop_for_plain_text() -> None:
    upstream = [
        TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id="m1"),
        TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT, message_id="m1", delta="Hello "
        ),
        TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT, message_id="m1", delta="world."
        ),
        TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id="m1"),
    ]
    out = await _drain(upstream)
    assert [e.type for e in out] == [
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
    ]
    # The first text run preserves the upstream message id.
    assert all(e.message_id == "m1" for e in out)


@pytest.mark.asyncio
async def test_rewrite_passes_through_native_tool_events() -> None:
    # Native tool-calling models already emit TOOL_CALL_* events; the rewriter
    # must not touch them.
    from ag_ui.core import ToolCallArgsEvent
    from ag_ui.core import ToolCallEndEvent
    from ag_ui.core import ToolCallStartEvent

    upstream = [
        ToolCallStartEvent(type=EventType.TOOL_CALL_START, tool_call_id="c1", tool_call_name="f"),
        ToolCallArgsEvent(type=EventType.TOOL_CALL_ARGS, tool_call_id="c1", delta="{}"),
        ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id="c1"),
    ]
    out = await _drain(upstream)
    assert [e.type for e in out] == [
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_ARGS,
        EventType.TOOL_CALL_END,
    ]
    assert all(e.tool_call_id == "c1" for e in out)
