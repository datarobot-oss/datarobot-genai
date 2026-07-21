# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NAT middleware: normalize native-NAT agent output to ``DRAgentEventResponse``.

Native NAT agents (e.g. ``tool_calling_agent`` / ``per_user_tool_calling_agent``) emit
``str`` for single output and ``ChatResponseChunk`` for streaming output. DRAgent's frontend,
moderation, and converters all expect the canonical ``DRAgentEventResponse``. This middleware
sits innermost (declared last in a function's ``middleware`` list) so it converts native output
into ``DRAgentEventResponse`` before any outer middleware (moderation, otel conventions) or the
frontend sees it. Agents that already emit ``DRAgentEventResponse`` pass through unchanged.

Because NAT middleware is per-function and opt-in (not inherited from the parent workflow), this
middleware must be declared on whichever function actually produces native-NAT output — including
an inner ``per_user_tool_calling_agent`` referenced by a memory wrapper's ``inner_agent_name``.

Streaming conversion from ``ChatResponseChunk`` to AG-UI events is implemented by
``convert_chunks_to_agui_events`` in this module (formerly ``dragent.frontends.stream_converter``).
"""

from __future__ import annotations

import contextlib
import logging
import sys
import uuid
from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from typing import Any
from typing import cast

from ag_ui.core import Event
from ag_ui.core import RunErrorEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from ag_ui.core import ToolCallArgsEvent
from ag_ui.core import ToolCallStartEvent
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_middleware
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.middleware import FunctionMiddlewareBaseConfig
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.middleware import CallNext
from nat.middleware.middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext

from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.dragent.frontends.converters import (
    convert_chat_response_to_dragent_event_response,
)
from datarobot_genai.dragent.frontends.converters import convert_str_to_dragent_text_response
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.frontends.tool_call_registry import mark_args_done
from datarobot_genai.dragent.frontends.tool_call_registry import register_tool_call

logger = logging.getLogger(__name__)


def resolve_streaming_tool_call_id(
    *,
    index: int,
    chunk_id: str | None,
    tool_index_map: dict[int, str],
) -> tuple[str | None, bool]:
    """Return ``(tool_call_id, is_new)`` for one OpenAI-style streaming tool delta.

    Follow-up chunks should only carry ``index``, but some providers (Gemini via
    LiteLLM) may re-emit a new ``id`` that appends a ``__thought__`` signature.
    Once an index is mapped, keep the first id for START/ARGS correlation.
    """
    if index in tool_index_map:
        return tool_index_map[index], False
    if chunk_id is None:
        return None, False
    return chunk_id, True


async def convert_chunks_to_agui_events(
    chunks: AsyncGenerator[ChatResponseChunk],
) -> AsyncGenerator[DRAgentEventResponse]:
    """Convert a ChatResponseChunk stream into AG-UI events.

    Yields ``DRAgentEventResponse`` batches as chunks arrive.  On upstream
    errors, emits ``RunErrorEvent`` and stops (does not propagate).  On
    ``GeneratorExit`` (client disconnect), exits silently.
    """
    active_message_id: str | None = None
    # parent_message_id for subsequent tool calls; a synthetic uuid here
    # renders an orphan message stub in the UI.
    last_text_message_id: str | None = None
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
                # Args streaming is complete for all tracked tool calls.
                # Flush any end/result events deferred by the step adaptor.
                for mapped_tc_id in tool_index_map.values():
                    events.extend(mark_args_done(mapped_tc_id))
                tool_index_map.clear()

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
                    tc_id, is_new = resolve_streaming_tool_call_id(
                        index=tc.index,
                        chunk_id=tc.id,
                        tool_index_map=tool_index_map,
                    )
                    if tc_id is None:
                        logger.warning(
                            "Tool call chunk at index %d has no id and no prior mapping; skipping",
                            tc.index,
                        )
                        continue
                    if is_new:
                        tool_index_map[tc.index] = tc_id
                        tool_name = tc.function.name if tc.function else ""
                        events.append(
                            ToolCallStartEvent(
                                tool_call_id=tc_id,
                                tool_call_name=tool_name,
                                parent_message_id=last_text_message_id or "",
                            )
                        )
                        # Hand the LLM-issued id to the step adaptor.
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
    # Mark remaining in-flight tool calls as args-done and flush deferred events.
    for mapped_tc_id in tool_index_map.values():
        end.extend(mark_args_done(mapped_tc_id))
    if active_message_id is not None:
        end.append(TextMessageEndEvent(message_id=active_message_id))
    if error is not None:
        end.append(RunErrorEvent(message=str(error), code="STREAM_ERROR"))
    if end:
        yield DRAgentEventResponse(events=end, usage_metrics=zero)


class DataRobotDRAgentNormalizationConfig(
    FunctionMiddlewareBaseConfig,  # type: ignore[misc]
    name="datarobot_dragent_normalization",  # type: ignore[call-arg]
):
    """NAT middleware: normalize native-NAT agent output to ``DRAgentEventResponse``.

    No configuration fields; declare it on a function's ``middleware`` list as the last
    (innermost) entry so downstream moderation and converters only ever see
    ``DRAgentEventResponse``.
    """


def _normalize_single_output(output: Any) -> Any:
    if isinstance(output, DRAgentEventResponse):
        return output
    if isinstance(output, ChatResponse):
        return convert_chat_response_to_dragent_event_response(output)
    if isinstance(output, str):
        return convert_str_to_dragent_text_response(output)
    return output


class DataRobotDRAgentNormalizationMiddleware(FunctionMiddleware):
    """Convert native-NAT output (``str`` / ``ChatResponse`` / ``ChatResponseChunk``) to
    ``DRAgentEventResponse``; pass ``DRAgentEventResponse`` through unchanged.
    """

    def __init__(
        self,
        config: DataRobotDRAgentNormalizationConfig,
        builder: Builder,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self._config = config

    @property
    def enabled(self) -> bool:
        return True

    async def function_middleware_invoke(
        self,
        *args: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext,  # noqa: ARG002
        **kwargs: Any,
    ) -> Any:
        output = await call_next(*args, **kwargs)
        return _normalize_single_output(output)

    async def function_middleware_stream(
        self,
        *args: Any,
        call_next: CallNextStream,
        context: FunctionMiddlewareContext,  # noqa: ARG002
        **kwargs: Any,
    ) -> AsyncIterator[DRAgentEventResponse]:
        async with contextlib.aclosing(
            cast(AsyncGenerator[Any, None], call_next(*args, **kwargs))
        ) as upstream:
            iterator = upstream.__aiter__()
            try:
                first = await iterator.__anext__()
            except StopAsyncIteration:
                return

            # dragent-native agents already emit DRAgentEventResponse -> passthrough.
            if isinstance(first, DRAgentEventResponse):
                yield first
                async for chunk in iterator:
                    yield chunk
                return

            # Native NAT agents emit ChatResponseChunk -> convert to AG-UI events. Reuse
            # the stateful stream converter, re-prepending the peeked first chunk.
            async def _rechained() -> AsyncGenerator[Any]:
                yield first
                async for chunk in iterator:
                    yield chunk

            async for event_response in convert_chunks_to_agui_events(_rechained()):
                yield event_response


@register_middleware(  # type: ignore[untyped-decorator]
    config_type=DataRobotDRAgentNormalizationConfig
)
async def datarobot_dragent_normalization_middleware(
    config: DataRobotDRAgentNormalizationConfig,
    builder: Builder,
) -> AsyncIterator[DataRobotDRAgentNormalizationMiddleware]:
    """Register the DRAgent output-normalization middleware for NAT workflows."""
    yield DataRobotDRAgentNormalizationMiddleware(config, builder)
