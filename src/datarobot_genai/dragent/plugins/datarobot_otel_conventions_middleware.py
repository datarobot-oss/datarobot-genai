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

import logging
from collections.abc import AsyncIterator
from typing import Any

from ag_ui.core import EventType
from ag_ui.core import RunAgentInput
from ag_ui.core import ToolCallStartEvent
from ag_ui.core import UserMessage
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_middleware
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import UserMessageContentRoleType
from nat.data_models.middleware import FunctionMiddlewareBaseConfig
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.middleware import CallNext
from nat.middleware.middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext
from opentelemetry import trace

from datarobot_genai.core.telemetry.nat_context import use_nat_workflow_trace_context
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)

# Parent span created per invocation so the Tracing table attributes always
# have a span to live on, even though NAT builds its own (non-SDK) spans.
AGENT_SPAN_NAME = "datarobot_agent"

# Span attributes that map to deployment Tracing table columns.
GEN_AI_PROMPT = "gen_ai.prompt"  # Prompt column
GEN_AI_COMPLETION = "gen_ai.completion"  # Completion column
TOOL_NAME = "tool_name"  # Tools column

# AG-UI event types that carry assistant text deltas.
_TEXT_EVENT_TYPES = (EventType.TEXT_MESSAGE_CONTENT, EventType.TEXT_MESSAGE_CHUNK)


def _last_user_message_content(value: Any) -> str | None:
    """Return the last user message content from a supported agent input.

    Only NAT's ``ChatRequestOrMessage`` and AG-UI's ``RunAgentInput`` are
    handled; any other input type returns ``None`` (no attribute set).
    """
    if isinstance(value, ChatRequestOrMessage):
        return _nat_last_user_message_content(value)
    if isinstance(value, RunAgentInput):
        return _ag_ui_last_user_message_content(value)
    return None


def _nat_last_user_message_content(request: ChatRequestOrMessage) -> str | None:
    for message in reversed(request.messages or []):
        if message.role == UserMessageContentRoleType.USER:
            return None if message.content is None else str(message.content)
    if request.input_message is not None:
        return request.input_message
    return None


def _ag_ui_last_user_message_content(run_agent_input: RunAgentInput) -> str | None:
    for message in reversed(run_agent_input.messages):
        if isinstance(message, UserMessage):
            return None if message.content is None else str(message.content)
    return None


def _response_text(response: DRAgentEventResponse) -> str:
    """Join assistant text deltas from a DRAgentEventResponse's AG-UI events."""
    return "".join(event.delta for event in response.events if event.type in _TEXT_EVENT_TYPES)


def _emit_tool_call_spans(response: DRAgentEventResponse) -> None:
    """Emit a short-lived span carrying ``tool_name`` for each tool-call start.

    NAT reports tool execution via intermediate-step end events we can't wrap,
    but it does surface a ``ToolCallStartEvent``. Creating and immediately
    ending a span with the ``tool_name`` attribute is enough to populate the
    Tracing table Tools column.
    """
    for event in response.events:
        if isinstance(event, ToolCallStartEvent):
            with tracer.start_as_current_span(event.tool_call_name) as span:
                span.set_attribute(TOOL_NAME, event.tool_call_name)


class DataRobotOtelConventionsMiddlewareConfig(
    FunctionMiddlewareBaseConfig,  # type: ignore[misc]
    name="datarobot_otel_conventions",  # type: ignore[call-arg]
):
    """DataRobot Open Telemetry Conventions:
    https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-tracing-code.html#map-spans-and-attributes-to-the-tracing-table.
    """


class DataRobotOtelConventionsMiddleware(
    FunctionMiddleware,  # type: ignore[misc]
):
    """DataRobot Open Telemetry Conventions middleware for DRAgent NAT workflows.

    Each invocation is wrapped in a dedicated ``datarobot_agent`` SDK span that
    carries the Tracing table attributes: the last user message becomes
    ``gen_ai.prompt`` and the workflow output becomes ``gen_ai.completion``.
    NAT builds its own (non-SDK) spans and may not open an OTel parent span, so
    we create our own to guarantee the attributes have a recording span to live
    on. Tool-call spans are emitted as children. The streaming path is
    reimplemented so text deltas can be aggregated across chunks within a single
    invocation.
    """

    def __init__(self, config: DataRobotOtelConventionsMiddlewareConfig, builder: Builder) -> None:  # noqa: ARG002
        super().__init__()

    @staticmethod
    def _prompt_from_args(args: tuple[Any, ...]) -> str | None:
        return _last_user_message_content(args[0]) if args else None

    @staticmethod
    def _completion_from_output(output: Any) -> str | None:
        # NAT non-streaming returns a plain str; every other path returns a
        # single aggregated DRAgentEventResponse.
        if isinstance(output, str):
            return output
        if isinstance(output, DRAgentEventResponse):
            return _response_text(output)
        return None

    async def function_middleware_invoke(
        self,
        *args: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext,  # noqa: ARG002
        **kwargs: Any,
    ) -> Any:
        with (
            use_nat_workflow_trace_context(),
            tracer.start_as_current_span(AGENT_SPAN_NAME) as span,
        ):
            prompt = self._prompt_from_args(args)
            if prompt is not None:
                span.set_attribute(GEN_AI_PROMPT, prompt)
            output = await call_next(*args, **kwargs)
            if isinstance(output, DRAgentEventResponse):
                _emit_tool_call_spans(output)
            completion = self._completion_from_output(output)
            if completion is not None:
                span.set_attribute(GEN_AI_COMPLETION, completion)
            return output

    async def function_middleware_stream(
        self,
        *args: Any,
        call_next: CallNextStream,
        context: FunctionMiddlewareContext,  # noqa: ARG002
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        with (
            use_nat_workflow_trace_context(),
            tracer.start_as_current_span(AGENT_SPAN_NAME) as span,
        ):
            prompt = self._prompt_from_args(args)
            if prompt is not None:
                span.set_attribute(GEN_AI_PROMPT, prompt)
            # Per-invocation accumulator; no cross-session state to manage.
            parts: list[str] = []
            try:
                async for chunk in call_next(*args, **kwargs):
                    if isinstance(chunk, DRAgentEventResponse):
                        _emit_tool_call_spans(chunk)
                        text = _response_text(chunk)
                        if text:
                            parts.append(text)
                    yield chunk
            finally:
                # Attach the completion in ``finally`` so it survives early teardown.
                # Downstream moderation may stop consuming and ``aclose()`` this generator
                # (throwing ``GeneratorExit`` at the ``yield``) before the loop exits normally
                # — e.g. when the moderation stream finishes before draining its source. Setting
                # the attribute after the loop would then be skipped, dropping ``gen_ai.completion``
                # even though the deltas were already seen. The span is still open here because the
                # enclosing ``with`` block outlives this ``finally``.
                if parts:
                    span.set_attribute(GEN_AI_COMPLETION, "".join(parts))


@register_middleware(  # type: ignore[untyped-decorator]
    config_type=DataRobotOtelConventionsMiddlewareConfig
)
async def datarobot_otel_conventions_middleware(
    config: DataRobotOtelConventionsMiddlewareConfig,
    builder: Builder,  # noqa: ARG001
) -> AsyncIterator[DataRobotOtelConventionsMiddleware]:
    """Register DataRobot Open Telemetry Conventions middleware for NAT/DRAgent workflows."""
    yield DataRobotOtelConventionsMiddleware(config, builder)
