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

"""Per-user variant of the NAT tool_calling_agent workflow.

The built-in ``tool_calling_agent`` is registered as a *shared* workflow, which
means NAT's dependency validator forbids it from referencing per-user function
groups such as ``a2a_client``.  This module registers an identical workflow under
the name ``per_user_tool_calling_agent`` using ``register_per_user_function`` so
that per-user function groups can be used while still benefiting from OpenAI-style
structured tool calling (``bind_tools``).

NAT 1.6 added a ``stream_fn`` that yields ``ChatResponseChunk``.  We wrap it
using ``DRAgentNestedReasoningStepAdaptor.process_chunks()`` to produce
``DRAgentEventResponse`` with valid AG-UI event sequences.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from collections.abc import Sequence
from typing import Any

from ag_ui.core import Event
from ag_ui.core import ToolCallEndEvent
from ag_ui.core import ToolCallResultEvent
from ag_ui.core import ToolCallStartEvent
from langchain_core.messages import BaseMessage
from langgraph.pregel._messages import StreamMessagesHandler
from langgraph.pregel._messages import _state_values
from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponse
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig
from nat.plugins.langchain.agent.tool_calling_agent.register import tool_calling_agent_workflow
from nat.utils.type_converter import GlobalTypeConverter

from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.frontends.stream_converter import convert_chunks_to_agui_events

logger = logging.getLogger(__name__)

# Workaround: prior assistant messages from chat history were leaking back into
# the response stream as a single trailing mega-chunk after the new response.
#
# How that happens:
#   1. NAT's `Message` data model has no `id` field, so when `_stream_fn`
#      converts the request into BaseMessages via
#      `trim_messages([m.model_dump() for m in chat.messages], ...)`, every
#      BaseMessage handed to the graph has `id=None`.
#   2. Langgraph's `StreamMessagesHandler` uses a `seen` set, keyed by
#      `message.id`, to avoid emitting the same message twice. Its
#      `on_chain_start` only records ids when `id is not None`, so id-less
#      inputs are never added to `seen`.
#   3. When `agent_node` returns, `on_chain_end` walks the new state and emits
#      every BaseMessage not in `seen`. Prior history items (id=None) match
#      that condition and get streamed to the client as if the model had just
#      produced them, even though they were part of the conversation history.
#
# Fix: assign a uuid to every id-less input message before the original
# `on_chain_start` runs, mirroring what `StreamMessagesHandler._emit` already
# does for output messages. This makes the seen set track them correctly, so
# `on_chain_end` no longer treats them as new output.
#
_original_on_chain_start = StreamMessagesHandler.on_chain_start


def _patched_on_chain_start(
    self: StreamMessagesHandler,
    serialized: dict[str, Any],
    inputs: dict[str, Any],
    **kwargs: Any,
) -> Any:
    for value in _state_values(inputs):
        if isinstance(value, BaseMessage) and value.id is None:
            value.id = str(uuid.uuid4())
        elif isinstance(value, Sequence) and not isinstance(value, str):
            for item in value:
                if isinstance(item, BaseMessage) and item.id is None:
                    item.id = str(uuid.uuid4())
    return _original_on_chain_start(self, serialized, inputs, **kwargs)


StreamMessagesHandler.on_chain_start = _patched_on_chain_start  # type: ignore[method-assign]


class PerUserToolCallAgentWorkflowConfig(
    ToolCallAgentWorkflowConfig,
    name="per_user_tool_calling_agent",  # type: ignore[call-arg]
):
    """Per-user version of tool_calling_agent."""

    pass


class _ToolCallResultBridge:
    """Correlate NAT tool completions back to AG-UI tool-call events.

    NAT's intermediate-step events do not carry the LLM's original
    ``tool_call_id``. This bridge reconstructs that mapping from streamed
    ``ToolCallStartEvent`` batches, then emits ``ToolCallResultEvent`` in
    ``END -> RESULT`` order while suppressing duplicate end events.
    """

    def __init__(self) -> None:
        self.pending_tool_call_ids_by_name: dict[str, list[str]] = {}
        self.pending_invocation_uuids_by_name: dict[str, list[str]] = {}
        self.pending_result_content_by_invocation_uuid: dict[str, str] = {}
        self.tool_call_id_by_invocation_uuid: dict[str, str] = {}
        self.ended_tool_call_ids: set[str] = set()
        self.queued_responses: asyncio.Queue[DRAgentEventResponse] = asyncio.Queue()
        self.zero_metrics = default_usage_metrics()

    def _queue_result(self, tool_call_id: str, content: str) -> None:
        self.queued_responses.put_nowait(
            DRAgentEventResponse(
                events=[
                    ToolCallResultEvent(
                        message_id=tool_call_id,
                        tool_call_id=tool_call_id,
                        content=content,
                        role="tool",
                    )
                ],
                usage_metrics=self.zero_metrics,
            )
        )

    def _flush_buffered_result_for_tool_call_id(
        self, invocation_uuid: str, tool_call_id: str
    ) -> None:
        content = self.pending_result_content_by_invocation_uuid.pop(invocation_uuid, None)
        if content is not None:
            self._queue_result(tool_call_id, content)
            return
        self.tool_call_id_by_invocation_uuid[invocation_uuid] = tool_call_id

    def _bind_streamed_tool_call_start(self, tool_call_name: str, tool_call_id: str) -> None:
        pending_invocations = self.pending_invocation_uuids_by_name.get(tool_call_name)
        if pending_invocations:
            invocation_uuid = pending_invocations.pop(0)
            self._flush_buffered_result_for_tool_call_id(invocation_uuid, tool_call_id)
            return

        self.pending_tool_call_ids_by_name.setdefault(tool_call_name, []).append(tool_call_id)

    def on_step(self, step: IntermediateStep) -> None:
        """Capture tool completions from NAT's intermediate-step stream."""
        payload = step.payload
        tool_name = payload.name or ""

        if payload.event_type in (
            IntermediateStepType.TOOL_START,
            IntermediateStepType.FUNCTION_START,
        ):
            pending_ids = self.pending_tool_call_ids_by_name.get(tool_name)
            if pending_ids:
                self._flush_buffered_result_for_tool_call_id(payload.UUID, pending_ids.pop(0))
            else:
                self.pending_invocation_uuids_by_name.setdefault(tool_name, []).append(payload.UUID)
            return

        if payload.event_type not in (
            IntermediateStepType.TOOL_END,
            IntermediateStepType.FUNCTION_END,
        ):
            return

        output = getattr(payload.metadata, "tool_outputs", None)
        if output is None and payload.data is not None:
            output = getattr(payload.data, "output", None)

        try:
            content = GlobalTypeConverter.get().convert(output, str)
        except Exception:
            content = str(output) if output is not None else ""

        tool_call_id = self.tool_call_id_by_invocation_uuid.pop(payload.UUID, None)
        if tool_call_id is not None:
            self._queue_result(tool_call_id, content)
            return

        pending_invocations = self.pending_invocation_uuids_by_name.get(tool_name)
        if pending_invocations and payload.UUID in pending_invocations:
            self.pending_result_content_by_invocation_uuid[payload.UUID] = content
            return

        # Fallback if the matching START happened before we subscribed.
        pending_ids = self.pending_tool_call_ids_by_name.get(tool_name)
        if pending_ids:
            self._queue_result(pending_ids.pop(0), content)
            return

    def emit_queued_results(self) -> list[DRAgentEventResponse]:
        """Emit any queued results, synthesizing END before RESULT if needed."""
        responses: list[DRAgentEventResponse] = []

        while not self.queued_responses.empty():
            response = self.queued_responses.get_nowait()
            prefix_events: list[Event] = []

            for event in response.events:
                if (
                    isinstance(event, ToolCallResultEvent)
                    and event.tool_call_id not in self.ended_tool_call_ids
                ):
                    prefix_events.append(ToolCallEndEvent(tool_call_id=event.tool_call_id))
                    self.ended_tool_call_ids.add(event.tool_call_id)

            if prefix_events:
                responses.append(
                    DRAgentEventResponse(
                        events=prefix_events,
                        usage_metrics=self.zero_metrics,
                    )
                )
            responses.append(response)

        return responses

    def _filter_stream_events(self, events: list[Event]) -> list[Event]:
        """Track start/end state and suppress duplicate END events."""
        filtered_events: list[Event] = []

        for event in events:
            if isinstance(event, ToolCallStartEvent):
                self._bind_streamed_tool_call_start(event.tool_call_name, event.tool_call_id)
            elif isinstance(event, ToolCallEndEvent):
                if event.tool_call_id in self.ended_tool_call_ids:
                    continue
                self.ended_tool_call_ids.add(event.tool_call_id)

            filtered_events.append(event)

        return filtered_events

    def rewrite_response(self, response: DRAgentEventResponse) -> list[DRAgentEventResponse]:
        """Insert queued results after the last END in a streamed batch."""
        filtered_events = self._filter_stream_events(response.events)
        if not filtered_events:
            return self.emit_queued_results()

        if filtered_events != response.events:
            response = response.model_copy(update={"events": filtered_events})

        last_end_index = next(
            (
                i
                for i in range(len(filtered_events) - 1, -1, -1)
                if isinstance(filtered_events[i], ToolCallEndEvent)
            ),
            None,
        )

        if last_end_index is None:
            return [*self.emit_queued_results(), response]

        split_index = last_end_index + 1
        rewritten = [
            response.model_copy(update={"events": filtered_events[:split_index]}),
            *self.emit_queued_results(),
        ]
        if split_index < len(filtered_events):
            rewritten.append(response.model_copy(update={"events": filtered_events[split_index:]}))
        return rewritten


async def _per_user_tool_calling_agent(
    config: PerUserToolCallAgentWorkflowConfig, builder: Any
) -> AsyncGenerator[Any, None]:
    """Wrap the original tool_calling_agent with AG-UI stream conversion."""
    from nat.builder.function_info import FunctionInfo  # noqa: PLC0415

    original_gen = tool_calling_agent_workflow.__wrapped__(config, builder)
    try:
        fn_info: FunctionInfo = await original_gen.__anext__()

        if fn_info.stream_fn is None:
            yield fn_info
            return

        original_stream_fn = fn_info.stream_fn

        async def wrapped_stream(
            chat_request_or_message: ChatRequestOrMessage,
        ) -> AsyncGenerator[DRAgentEventResponse, None]:
            bridge = _ToolCallResultBridge()

            subscription = None
            try:
                subscription = Context.get().intermediate_step_manager.subscribe(
                    on_next=bridge.on_step
                )
            except Exception as exc:
                logger.warning("Failed to subscribe to intermediate steps: %s", exc)

            try:
                async for event in convert_chunks_to_agui_events(
                    original_stream_fn(chat_request_or_message)
                ):
                    for rewritten in bridge.rewrite_response(event):
                        yield rewritten

                for rewritten in bridge.emit_queued_results():
                    yield rewritten
            finally:
                if subscription is not None:
                    try:
                        subscription.unsubscribe()
                    except Exception as exc:
                        logger.warning("Failed to unsubscribe intermediate steps: %s", exc)

        yield FunctionInfo.create(
            single_fn=fn_info.single_fn,
            stream_fn=wrapped_stream,
            description=fn_info.description,
        )
    finally:
        await original_gen.aclose()


register_per_user_function(
    config_type=PerUserToolCallAgentWorkflowConfig,
    input_type=ChatRequest,
    single_output_type=ChatResponse,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)(_per_user_tool_calling_agent)
