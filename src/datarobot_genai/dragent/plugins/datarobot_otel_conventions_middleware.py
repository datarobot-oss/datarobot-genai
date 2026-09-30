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
from ag_ui.core import TextMessageEndEvent
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

from datarobot_genai.core.agents import default_usage_metrics
from datarobot_genai.core.agents import track_open_text_in_events
from datarobot_genai.core.telemetry.agent_span import AGENT_SPAN_NAME
from datarobot_genai.core.telemetry.agent_span import ERROR_TYPE
from datarobot_genai.core.telemetry.agent_span import GEN_AI_COMPLETION
from datarobot_genai.core.telemetry.agent_span import GEN_AI_PROMPT
from datarobot_genai.core.telemetry.agent_span import agent_span
from datarobot_genai.core.telemetry.agent_span import last_user_message
from datarobot_genai.core.telemetry.nat_context import use_nat_workflow_trace_context
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.frontends.response import run_error_response

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)

# AG-UI event types that carry assistant text deltas.
_TEXT_EVENT_TYPES = (EventType.TEXT_MESSAGE_CONTENT, EventType.TEXT_MESSAGE_CHUNK)

# Re-exported: the span/attribute names now live in the NAT-free
# ``core.telemetry.agent_span`` so non-NAT callers can share them.
__all__ = [
    "AGENT_SPAN_NAME",
    "ERROR_TYPE",
    "GEN_AI_COMPLETION",
    "GEN_AI_PROMPT",
    "DataRobotOtelConventionsMiddleware",
    "DataRobotOtelConventionsMiddlewareConfig",
]


def _thread_id_from_args(args: tuple[Any, ...]) -> str | None:
    """Return the AG-UI ``thread_id`` from a supported agent input, if present.

    Only ``RunAgentInput`` (AG-UI) carries a thread id; NAT's own
    ``ChatRequestOrMessage`` has no equivalent concept, so any other input
    type returns ``None`` (no attribute set) - matching ``_last_user_message_content``.
    """
    value = args[0] if args else None
    return value.thread_id if isinstance(value, RunAgentInput) else None


def _last_user_message_content(value: Any) -> str | None:
    """Return the last user message content from a supported agent input.

    Only NAT's ``ChatRequestOrMessage`` and AG-UI's ``RunAgentInput`` are
    handled; any other input type returns ``None`` (no attribute set).
    """
    if isinstance(value, ChatRequestOrMessage):
        return _nat_last_user_message_content(value)
    if isinstance(value, RunAgentInput):
        return last_user_message(value)
    return None


def _nat_last_user_message_content(request: ChatRequestOrMessage) -> str | None:
    for message in reversed(request.messages or []):
        if message.role == UserMessageContentRoleType.USER:
            return None if message.content is None else str(message.content)
    if request.input_message is not None:
        return request.input_message
    return None


def _response_text(response: DRAgentEventResponse) -> str:
    """Join assistant text deltas from a DRAgentEventResponse's AG-UI events."""
    return "".join(event.delta for event in response.events if event.type in _TEXT_EVENT_TYPES)


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
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> Any:
        with (
            use_nat_workflow_trace_context(),
            agent_span(
                context.name,
                prompt=self._prompt_from_args(args),
                session_id=_thread_id_from_args(args),
                tracer=tracer,
            ) as recorder,
        ):
            output = await call_next(*args, **kwargs)
            if isinstance(output, DRAgentEventResponse):
                recorder.observe(output.events)
            completion = self._completion_from_output(output)
            if completion is not None:
                recorder.set_completion(completion)
            return output

    async def function_middleware_stream(
        self,
        *args: Any,
        call_next: CallNextStream,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        # ``agent_span`` writes the aggregated completion in its own ``finally``,
        # so it survives early teardown: downstream moderation may stop consuming
        # and ``aclose()`` this generator (throwing ``GeneratorExit`` at the
        # ``yield``) before the loop exits normally.
        with (
            use_nat_workflow_trace_context(),
            agent_span(
                context.name,
                prompt=self._prompt_from_args(args),
                session_id=_thread_id_from_args(args),
                tracer=tracer,
            ) as recorder,
        ):
            open_text_ids: set[str] = set()
            try:
                async for chunk in call_next(*args, **kwargs):
                    if isinstance(chunk, DRAgentEventResponse):
                        recorder.observe(chunk.events)
                        track_open_text_in_events(open_text_ids, chunk.events)
                    yield chunk
            except Exception as exc:
                # Close open text segments, then end the run with a terminal RUN_ERROR instead of
                # propagating (matches the moderation middleware's failure path).
                logger.exception("Agent stream failed")
                for message_id in open_text_ids:
                    yield DRAgentEventResponse(
                        events=[TextMessageEndEvent(message_id=message_id)],
                        usage_metrics=default_usage_metrics(),
                    )
                error_response = run_error_response(str(exc))
                recorder.observe(error_response.events)
                yield error_response


@register_middleware(  # type: ignore[untyped-decorator]
    config_type=DataRobotOtelConventionsMiddlewareConfig
)
async def datarobot_otel_conventions_middleware(
    config: DataRobotOtelConventionsMiddlewareConfig,
    builder: Builder,  # noqa: ARG001
) -> AsyncIterator[DataRobotOtelConventionsMiddleware]:
    """Register DataRobot Open Telemetry Conventions middleware for NAT/DRAgent workflows."""
    yield DataRobotOtelConventionsMiddleware(config, builder)
