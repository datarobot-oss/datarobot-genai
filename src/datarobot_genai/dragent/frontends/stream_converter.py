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

"""Convert ChatResponseChunk streams into AG-UI events.

NAT 1.6 added a ``stream_fn`` to ``tool_calling_agent`` that yields
``ChatResponseChunk`` objects (OpenAI-compatible streaming deltas).  This
module converts that stream into ``DRAgentEventResponse`` batches containing
AG-UI ``TextMessage*`` and ``ToolCall*`` events.

See docs/nat-1.6-streaming.md for the full design.
"""

import logging
import sys
import uuid
from collections.abc import AsyncGenerator

from ag_ui.core import Event
from ag_ui.core import RunErrorEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallStartEvent
from nat.data_models.api_server import ChatResponseChunk

from datarobot_genai.core.agents import default_usage_metrics

from .response import DRAgentEventResponse
from .tool_call_registry import register_tool_call

logger = logging.getLogger(__name__)


async def convert_chunks_to_agui_events(
    chunks: AsyncGenerator[ChatResponseChunk, None],
) -> AsyncGenerator[DRAgentEventResponse, None]:
    """Convert a ChatResponseChunk stream into AG-UI events.

    Yields ``DRAgentEventResponse`` batches as chunks arrive.  On upstream
    errors, emits ``RunErrorEvent`` and stops (does not propagate).  On
    ``GeneratorExit`` (client disconnect), exits silently.
    """
    active_message_id: str | None = None
    # parent_message_id of subsequent tool calls; threads them under the
    # assistant message that issued them. A synthetic uuid here renders
    # an orphan message stub in the UI.
    last_text_message_id: str | None = None
    active_tool_calls: set[str] = set()
    seen_tool_calls: bool = False
    tool_index_map: dict[int, str] = {}
    zero = default_usage_metrics()

    error: Exception | None = None
    try:
        async for chunk in chunks:
            if not isinstance(chunk, ChatResponseChunk) or not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            events: list[Event] = []

            if delta and delta.content:
                # Step adaptor owns ToolCallEnd at FUNCTION_END / TOOL_END;
                # emitting End here would fire after Result and strand the UI.
                active_tool_calls.clear()

                if active_message_id is None:
                    # After a tool call cycle, the LLM response is a new turn
                    # but chunk.id may be reused.  Force a unique messageId.
                    active_message_id = (
                        str(uuid.uuid4()) if seen_tool_calls else (chunk.id or str(uuid.uuid4()))
                    )
                    events.append(TextMessageStartEvent(message_id=active_message_id))
                events.append(
                    TextMessageContentEvent(message_id=active_message_id, delta=delta.content)
                )

            if delta and delta.tool_calls:
                # Close any active text message before starting tool calls.
                if active_message_id is not None:
                    events.append(TextMessageEndEvent(message_id=active_message_id))
                    last_text_message_id = active_message_id
                    active_message_id = None
                seen_tool_calls = True

                for tc in delta.tool_calls:
                    tc_id = tc.id or tool_index_map.get(tc.index)  # type: ignore[assignment]
                    if tc_id is None:
                        logger.warning(
                            "Tool call chunk at index %d has no id and no prior mapping; skipping",
                            tc.index,
                        )
                        continue
                    is_new = tc.id is not None and tc_id not in active_tool_calls
                    if is_new:
                        active_tool_calls.add(tc_id)
                        tool_index_map[tc.index] = tc_id
                        tool_name = tc.function.name if tc.function else ""
                        events.append(
                            ToolCallStartEvent(
                                tool_call_id=tc_id,
                                tool_call_name=tool_name,
                                parent_message_id=last_text_message_id or "",
                            )
                        )
                        # Hand the LLM-issued id to the step adaptor; it
                        # binds ToolCallResult to it on FUNCTION_END.
                        if tool_name:
                            register_tool_call(tool_name, tc_id)
                    arguments = tc.function.arguments if tc.function else None
                    if arguments:
                        events.append(ToolCallArgsEvent(tool_call_id=tc_id, delta=arguments))

            if events:
                yield DRAgentEventResponse(events=events, usage_metrics=zero, original_chunk=chunk)
    except Exception as exc:
        error = exc
    finally:
        if sys.exc_info()[0] is GeneratorExit:
            logger.debug("Client disconnected before end events could be delivered")
            return

    # Emit end/error events after the stream completes (normally or on error).
    # Errors are surfaced to the AG-UI client via RunErrorEvent rather than
    # propagated as exceptions, so NAT's streaming infrastructure stays stable.
    end: list[Event] = []
    if active_message_id is not None:
        end.append(TextMessageEndEvent(message_id=active_message_id))
    for tc_id in active_tool_calls:
        end.append(ToolCallEndEvent(tool_call_id=tc_id))
    if error is not None:
        end.append(RunErrorEvent(message=str(error), code="STREAM_ERROR"))
    if end:
        yield DRAgentEventResponse(events=end, usage_metrics=zero)
