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

"""Convert NAT ChatResponseChunk stream to AG-UI event stream.

NAT 1.6's tool_calling_agent streams ChatResponseChunk objects.
The DRAGent frontend expects DRAgentEventResponse with valid AG-UI
event sequences (START before CONTENT, END at stream close).

This module converts between the two, tracking lifecycle state
per-request with no global state.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from ag_ui.core import EventType
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallStartEvent
from nat.data_models.api_server import ChatResponseChunk

from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse

logger = logging.getLogger(__name__)


async def convert_chunks_to_ag_ui_events(
    chunks: AsyncGenerator[ChatResponseChunk, None],
) -> AsyncGenerator[DRAgentEventResponse, None]:
    """Convert a ChatResponseChunk stream into DRAgentEventResponse with AG-UI lifecycle.

    Tracks active text messages and tool calls to emit proper
    START/CONTENT/END sequences. State is local to this call.
    """
    active_message_id: str | None = None
    active_tool_calls: list[str] = []
    # Map tool call index -> id for follow-up chunks that have id=None
    tool_call_index_to_id: dict[int, str] = {}
    zero = default_usage_metrics()

    try:
        async for chunk in chunks:
            if not isinstance(chunk, ChatResponseChunk) or not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            events: list[Any] = []

            # Text content
            if delta and delta.content:
                if active_message_id is None:
                    active_message_id = chunk.id or ""
                    events.append(
                        TextMessageStartEvent(
                            type=EventType.TEXT_MESSAGE_START,
                            message_id=active_message_id,
                            role="assistant",
                        )
                    )
                events.append(
                    TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=active_message_id,
                        delta=delta.content,
                    )
                )

            # Tool calls
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    # First chunk has id; follow-ups have id=None, use index to look up
                    tc_id = tc.id or tool_call_index_to_id.get(tc.index)
                    if tc_id is None:
                        logger.warning(
                            "Tool call chunk at index %d has no id and no prior mapping; skipping",
                            tc.index,
                        )
                        continue
                    if tc.id and tc_id not in active_tool_calls:
                        active_tool_calls.append(tc_id)
                        tool_call_index_to_id[tc.index] = tc_id
                        name = tc.function.name if tc.function else ""
                        events.append(
                            ToolCallStartEvent(
                                type=EventType.TOOL_CALL_START,
                                tool_call_id=tc_id,
                                tool_call_name=name,
                            )
                        )
                    if tc.function and tc.function.arguments:
                        events.append(
                            ToolCallArgsEvent(
                                type=EventType.TOOL_CALL_ARGS,
                                tool_call_id=tc_id,
                                delta=tc.function.arguments,
                            )
                        )

            if events:
                yield DRAgentEventResponse(events=events, usage_metrics=zero)
    finally:
        # Close any active lifecycle at stream end
        end: list[Any] = []
        if active_message_id is not None:
            end.append(
                TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=active_message_id)
            )
        for tc_id in active_tool_calls:
            end.append(ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id=tc_id))
        if end:
            try:
                yield DRAgentEventResponse(events=end, usage_metrics=zero)
            except GeneratorExit:
                logger.debug("Client disconnected before end events could be delivered")
