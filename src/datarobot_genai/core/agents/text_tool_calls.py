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
"""Recover Hermes-style tool-call markup from a streamed text channel.

Some models (notably Anthropic served via AWS Bedrock behind a text tool-calling
template) emit tool calls and their results as literal ``<tool_call>`` /
``<tool_response>`` markup inside the assistant *text* stream instead of as
native structured tool calls. Without recovery, that markup flows verbatim
through ``TEXT_MESSAGE_CONTENT`` and renders as raw text in AG-UI clients, even
though the tools execute upstream.

This module is framework-neutral: :class:`HermesTextToolCallParser` is a pure
incremental parser, and :func:`rewrite_text_tool_calls` transforms any agent's
AG-UI event stream (``BaseAgent.invoke`` output). Applying it once at the shared
chat-completion boundary covers every framework (LangGraph, CrewAI, LlamaIndex,
NAT) without per-framework changes. The transform is display-only: it never
re-executes tools, so a transcript whose tools already ran upstream renders
correctly.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Optional

from ag_ui.core import Event
from ag_ui.core import EventType
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallResultEvent
from ag_ui.core import ToolCallStartEvent

from datarobot_genai.core.agents.base import InvokeReturn
from datarobot_genai.core.agents.base import UsageMetrics

if TYPE_CHECKING:
    from ragas import MultiTurnSample

logger = logging.getLogger(__name__)

OPEN_CALL = "<tool_call>"
CLOSE_CALL = "</tool_call>"
OPEN_RESULT = "<tool_response>"
CLOSE_RESULT = "</tool_response>"


# --------------------------------------------------------------------------- #
# Pure incremental parser
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TextSegment:
    """Plain assistant prose that should stream through unchanged."""

    text: str


@dataclass(frozen=True)
class ToolCallSegment:
    """A recovered tool call. ``arguments`` is a JSON-encoded object string."""

    name: str
    arguments: str


@dataclass(frozen=True)
class ToolResultSegment:
    """A recovered tool result, correlated to the preceding tool call by order."""

    content: str


Segment = TextSegment | ToolCallSegment | ToolResultSegment


class HermesTextToolCallParser:
    """Incrementally extract ``<tool_call>``/``<tool_response>`` markup from text.

    Feed streamed text deltas via :meth:`feed`; call :meth:`flush` once the text
    stream ends. Markup may span delta boundaries: the parser buffers a minimal
    tail that could be the start of a tag and only emits text it is sure is plain
    prose. A ``<tool_call>`` body that is not valid JSON is passed through as text
    rather than dropped.
    """

    _OPENERS = (OPEN_CALL, OPEN_RESULT)

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[Segment]:
        """Consume a text delta and return any segments now fully resolved."""
        self._buffer += text
        return self._consume(final=False)

    def flush(self) -> list[Segment]:
        """Resolve and return everything still buffered at end of stream."""
        return self._consume(final=True)

    def _consume(self, *, final: bool) -> list[Segment]:
        segments: list[Segment] = []
        while True:
            open_idx, opener = self._next_opener()

            # No opener in the buffer: emit safe prose, holding back any tail that
            # could be the start of a tag (unless this is the final flush).
            if opener is None:
                self._emit_plain_tail(segments, final=final)
                return segments

            closer = CLOSE_CALL if opener == OPEN_CALL else CLOSE_RESULT
            close_idx = self._buffer.find(closer, open_idx + len(opener))

            # Opener seen but the block is not yet complete.
            if close_idx == -1:
                if open_idx > 0:
                    segments.append(TextSegment(self._buffer[:open_idx]))
                self._buffer = self._buffer[open_idx:]
                if not final:
                    return segments
                # Stream ended mid-block: surface the unterminated remainder as text.
                segments.append(TextSegment(self._buffer))
                self._buffer = ""
                return segments

            # Complete block: prose before it, then the parsed tool segment.
            if open_idx > 0:
                segments.append(TextSegment(self._buffer[:open_idx]))
            inner = self._buffer[open_idx + len(opener) : close_idx]
            segments.append(self._parse_block(opener, inner))
            self._buffer = self._buffer[close_idx + len(closer) :]

    def _next_opener(self) -> tuple[int, str | None]:
        best_idx = -1
        best_opener: str | None = None
        for opener in self._OPENERS:
            idx = self._buffer.find(opener)
            if idx == -1:
                continue
            if best_idx == -1 or idx < best_idx:
                best_idx = idx
                best_opener = opener
        return best_idx, best_opener

    def _emit_plain_tail(self, segments: list[Segment], *, final: bool) -> None:
        if final:
            if self._buffer:
                segments.append(TextSegment(self._buffer))
            self._buffer = ""
            return
        hold = self._partial_opener_suffix_len()
        emit_to = len(self._buffer) - hold
        if emit_to > 0:
            segments.append(TextSegment(self._buffer[:emit_to]))
        self._buffer = self._buffer[emit_to:]

    def _partial_opener_suffix_len(self) -> int:
        """Length of the longest buffer suffix that is a proper opener prefix."""
        longest = 0
        for opener in self._OPENERS:
            limit = min(len(opener) - 1, len(self._buffer))
            for length in range(limit, 0, -1):
                if self._buffer.endswith(opener[:length]):
                    longest = max(longest, length)
                    break
        return longest

    def _parse_block(self, opener: str, inner: str) -> Segment:
        if opener == OPEN_RESULT:
            return ToolResultSegment(inner.strip())
        return self._parse_tool_call(inner)

    def _parse_tool_call(self, inner: str) -> Segment:
        try:
            payload = json.loads(inner)
        except json.JSONDecodeError:
            logger.debug("Could not parse <tool_call> body as JSON: %r", inner)
            return TextSegment(f"{OPEN_CALL}{inner}{CLOSE_CALL}")
        if not isinstance(payload, dict):
            logger.debug("Unexpected <tool_call> payload type: %r", payload)
            return TextSegment(f"{OPEN_CALL}{inner}{CLOSE_CALL}")
        raw_args = payload.get("arguments", {})
        arguments = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)
        return ToolCallSegment(name=str(payload.get("name", "")), arguments=arguments)


# --------------------------------------------------------------------------- #
# AG-UI event stream transform
# --------------------------------------------------------------------------- #
# Tuples flowing through the AG-UI event stream: (event, ragas interactions, usage).
_StreamItem = tuple[Event, Optional["MultiTurnSample"], UsageMetrics]


@dataclass
class _RewriteState:
    """Mutable state shared by the rewrite helpers.

    ``source_id`` is the message id of the in-flight upstream text message;
    ``open_text_id`` is the id of a text message we have started but not yet
    ended; ``pending_call_ids`` correlates recovered tool calls to their results
    in arrival order (Hermes pairs a ``<tool_call>`` with the next
    ``<tool_response>``).
    """

    source_id: str = ""
    source_id_used: bool = False
    open_text_id: str | None = None
    pending_call_ids: list[str] = field(default_factory=list)


def _close_open_text(state: _RewriteState, usage: UsageMetrics) -> Iterator[_StreamItem]:
    if state.open_text_id is None:
        return
    yield (
        TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=state.open_text_id),
        None,
        usage,
    )
    state.open_text_id = None


def _emit_text(text: str, state: _RewriteState, usage: UsageMetrics) -> Iterator[_StreamItem]:
    if not text:
        return
    # Drop inter-tag whitespace rather than open an empty assistant text bubble.
    if state.open_text_id is None and not text.strip():
        return
    if state.open_text_id is None:
        # Keep the first text run on the upstream id so the common no-markup path is
        # byte-identical to the native flow; mint fresh ids for runs that resume after
        # a recovered tool call (reusing the ended id would collide on the client).
        if state.source_id and not state.source_id_used:
            state.open_text_id = state.source_id
            state.source_id_used = True
        else:
            state.open_text_id = uuid.uuid4().hex
        yield (
            TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=state.open_text_id,
                role="assistant",
            ),
            None,
            usage,
        )
    yield (
        TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id=state.open_text_id,
            delta=text,
        ),
        None,
        usage,
    )


def _emit_tool_call(
    segment: ToolCallSegment, state: _RewriteState, usage: UsageMetrics
) -> Iterator[_StreamItem]:
    yield from _close_open_text(state, usage)
    call_id = uuid.uuid4().hex
    state.pending_call_ids.append(call_id)
    yield (
        ToolCallStartEvent(
            type=EventType.TOOL_CALL_START,
            tool_call_id=call_id,
            tool_call_name=segment.name,
            parent_message_id=state.source_id,
        ),
        None,
        usage,
    )
    yield (
        ToolCallArgsEvent(
            type=EventType.TOOL_CALL_ARGS, tool_call_id=call_id, delta=segment.arguments
        ),
        None,
        usage,
    )
    yield (
        ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id=call_id),
        None,
        usage,
    )


def _emit_tool_result(
    segment: ToolResultSegment, state: _RewriteState, usage: UsageMetrics
) -> Iterator[_StreamItem]:
    yield from _close_open_text(state, usage)
    call_id = state.pending_call_ids.pop(0) if state.pending_call_ids else uuid.uuid4().hex
    yield (
        ToolCallResultEvent(
            type=EventType.TOOL_CALL_RESULT,
            message_id=call_id,
            tool_call_id=call_id,
            content=segment.content,
            role="tool",
        ),
        None,
        usage,
    )


def _segments_to_events(
    segments: list[Segment], state: _RewriteState, usage: UsageMetrics
) -> Iterator[_StreamItem]:
    for segment in segments:
        if isinstance(segment, TextSegment):
            yield from _emit_text(segment.text, state, usage)
            continue
        if isinstance(segment, ToolCallSegment):
            yield from _emit_tool_call(segment, state, usage)
            continue
        yield from _emit_tool_result(segment, state, usage)


async def rewrite_text_tool_calls(stream: InvokeReturn) -> InvokeReturn:
    """Recover text-encoded tool calls from any agent's AG-UI event stream.

    Wrap a ``BaseAgent.invoke`` stream so Hermes-style ``<tool_call>`` /
    ``<tool_response>`` markup arriving as ``TEXT_MESSAGE_CONTENT`` is re-emitted
    as the same structured ``TOOL_CALL_START``/``ARGS``/``END``/``RESULT`` events
    the native tool-calling path produces. Non-text events pass through untouched,
    so native tool calls (which already arrive as ``TOOL_CALL_*`` events, never
    text) are unaffected, and text without markup streams through unchanged.
    """
    parser = HermesTextToolCallParser()
    state = _RewriteState()
    async for event, interactions, usage in stream:
        if event.type == EventType.TEXT_MESSAGE_START:
            state.source_id = str(getattr(event, "message_id", "") or "")
            state.source_id_used = False
            continue
        if event.type == EventType.TEXT_MESSAGE_CONTENT:
            delta = str(getattr(event, "delta", "") or "")
            for item in _segments_to_events(parser.feed(delta), state, usage):
                yield item
            continue
        if event.type == EventType.TEXT_MESSAGE_END:
            for item in _segments_to_events(parser.flush(), state, usage):
                yield item
            for item in _close_open_text(state, usage):
                yield item
            state.source_id = ""
            continue
        # Any non-text event: close our open text message first to keep ordering valid.
        for item in _close_open_text(state, usage):
            yield item
        yield (event, interactions, usage)
