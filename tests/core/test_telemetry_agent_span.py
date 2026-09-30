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

"""Unit tests for the NAT-free ``agent_span`` Tracing-table annotations."""

from __future__ import annotations

import subprocess
import sys

import pytest
from ag_ui.core import AssistantMessage
from ag_ui.core import BaseMessage
from ag_ui.core import Message
from ag_ui.core import RunAgentInput
from ag_ui.core import RunErrorEvent
from ag_ui.core import StepStartedEvent
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import ToolCallStartEvent
from ag_ui.core import UserMessage
from datarobot_opentelemetry.semconv import SpanAttributes as DataRobotSpanAttributes
from opentelemetry import baggage
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.trace import Tracer

from datarobot_genai.core.telemetry.agent_identity import GEN_AI_AGENT_NAME_BAGGAGE_KEY
from datarobot_genai.core.telemetry.agent_span import AGENT_SPAN_NAME
from datarobot_genai.core.telemetry.agent_span import ERROR_TYPE
from datarobot_genai.core.telemetry.agent_span import GEN_AI_COMPLETION
from datarobot_genai.core.telemetry.agent_span import GEN_AI_PROMPT
from datarobot_genai.core.telemetry.agent_span import RUN_ERROR_CODE
from datarobot_genai.core.telemetry.agent_span import agent_span
from datarobot_genai.core.telemetry.agent_span import last_user_message


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter: InMemorySpanExporter) -> Tracer:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__)


def _spans(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    return {span.name: span for span in exporter.get_finished_spans()}


def _run_input(*messages: Message) -> RunAgentInput:
    return RunAgentInput(
        thread_id="thread",
        run_id="run",
        state={},
        messages=list(messages),
        tools=[],
        context=[],
        forwarded_props={},
    )


# ---------------------------------------------------------------------------
# last_user_message
# ---------------------------------------------------------------------------


def test_last_user_message_returns_last_user_message() -> None:
    # GIVEN a conversation ending in an assistant reply
    run_input = _run_input(
        UserMessage(id="u1", content="first"),
        UserMessage(id="u2", content="last"),
        AssistantMessage(id="a1", content="reply"),
    )

    # WHEN / THEN the last user turn is returned
    assert last_user_message(run_input) == "last"


def test_last_user_message_matches_generic_messages_by_role() -> None:
    # GIVEN history rebuilt as generic messages, as an application backend does
    run_input = _run_input()
    run_input.messages = [BaseMessage(id="u1", role="user", content="from history")]  # type: ignore[list-item]

    # WHEN / THEN the role, not the class, identifies the user turn
    assert last_user_message(run_input) == "from history"


def test_last_user_message_returns_none_without_user() -> None:
    # GIVEN no user messages
    run_input = _run_input(AssistantMessage(id="a1", content="reply"))

    # WHEN / THEN nothing is returned
    assert last_user_message(run_input) is None


# ---------------------------------------------------------------------------
# agent_span
# ---------------------------------------------------------------------------


def test_agent_span_sets_input_attributes(exporter: InMemorySpanExporter, tracer: Tracer) -> None:
    # GIVEN an agent invocation with a prompt and a session
    # WHEN it runs inside agent_span
    with agent_span("researcher", prompt="hi", session_id="thread", tracer=tracer):
        pass

    # THEN the agent span carries the Tracing table input attributes
    span = _spans(exporter)[AGENT_SPAN_NAME]
    assert span.attributes is not None
    assert span.attributes[DataRobotSpanAttributes.GEN_AI_AGENT_NAME] == "researcher"
    assert span.attributes[GEN_AI_PROMPT] == "hi"
    assert span.attributes[DataRobotSpanAttributes.DATAROBOT_SESSION_ID] == "thread"


def test_agent_span_omits_absent_prompt_and_session(
    exporter: InMemorySpanExporter, tracer: Tracer
) -> None:
    # GIVEN no prompt or session
    # WHEN the invocation runs
    with agent_span("researcher", tracer=tracer):
        pass

    # THEN neither attribute is set
    span = _spans(exporter)[AGENT_SPAN_NAME]
    assert span.attributes is not None
    assert GEN_AI_PROMPT not in span.attributes
    assert DataRobotSpanAttributes.DATAROBOT_SESSION_ID not in span.attributes


def test_agent_span_aggregates_completion_across_observe_calls(
    exporter: InMemorySpanExporter, tracer: Tracer
) -> None:
    # GIVEN text deltas arriving across several chunks, mixed with other events
    # WHEN each chunk is observed
    with agent_span("researcher", tracer=tracer) as recorder:
        recorder.observe([TextMessageContentEvent(message_id="m1", delta="Hello ")])
        recorder.observe([StepStartedEvent(step_name="step")])
        recorder.observe([TextMessageChunkEvent(message_id="m1", delta="world")])

    # THEN the completion is the joined text
    span = _spans(exporter)[AGENT_SPAN_NAME]
    assert span.attributes is not None
    assert span.attributes[GEN_AI_COMPLETION] == "Hello world"


def test_agent_span_without_text_sets_no_completion(
    exporter: InMemorySpanExporter, tracer: Tracer
) -> None:
    # GIVEN only non-text events
    # WHEN they are observed
    with agent_span("researcher", tracer=tracer) as recorder:
        recorder.observe([StepStartedEvent(step_name="step")])

    # THEN no completion is recorded
    span = _spans(exporter)[AGENT_SPAN_NAME]
    assert span.attributes is not None
    assert GEN_AI_COMPLETION not in span.attributes


def test_set_completion_overrides_aggregated_text(
    exporter: InMemorySpanExporter, tracer: Tracer
) -> None:
    # GIVEN observed deltas and an explicit completion
    # WHEN both are recorded
    with agent_span("researcher", tracer=tracer) as recorder:
        recorder.observe([TextMessageContentEvent(message_id="m1", delta="partial")])
        recorder.set_completion("final answer")

    # THEN the explicit completion wins
    span = _spans(exporter)[AGENT_SPAN_NAME]
    assert span.attributes is not None
    assert span.attributes[GEN_AI_COMPLETION] == "final answer"


def test_agent_span_emits_tool_spans_under_the_agent_span(
    exporter: InMemorySpanExporter, tracer: Tracer
) -> None:
    # GIVEN a tool call start event
    # WHEN it is observed
    with agent_span("researcher", tracer=tracer) as recorder:
        recorder.observe([ToolCallStartEvent(tool_call_id="tc1", tool_call_name="search")])

    # THEN a child span carries the tool and agent names
    spans = _spans(exporter)
    tool_span = spans["search"]
    assert tool_span.attributes is not None
    assert tool_span.attributes[DataRobotSpanAttributes.GEN_AI_TOOL_NAME] == "search"
    assert tool_span.attributes[DataRobotSpanAttributes.GEN_AI_AGENT_NAME] == "researcher"
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == spans[AGENT_SPAN_NAME].context.span_id


@pytest.mark.parametrize(
    ("code", "expected"), [("agent_failed", "agent_failed"), (None, RUN_ERROR_CODE)]
)
def test_agent_span_marks_error_on_run_error_event(
    exporter: InMemorySpanExporter, tracer: Tracer, code: str | None, expected: str
) -> None:
    # GIVEN a run error event, with or without a code
    # WHEN it is observed
    with agent_span("researcher", tracer=tracer) as recorder:
        recorder.observe([RunErrorEvent(message="boom", code=code)])

    # THEN the span is failed with the error type
    span = _spans(exporter)[AGENT_SPAN_NAME]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes is not None
    assert span.attributes[ERROR_TYPE] == expected


def test_agent_span_sets_completion_when_the_body_raises(
    exporter: InMemorySpanExporter, tracer: Tracer
) -> None:
    # GIVEN text already observed when the caller's stream is torn down
    # WHEN the body exits with an exception (e.g. GeneratorExit from aclose())
    with pytest.raises(GeneratorExit):
        with agent_span("researcher", tracer=tracer) as recorder:
            recorder.observe([TextMessageContentEvent(message_id="m1", delta="partial")])
            raise GeneratorExit

    # THEN the completion seen so far is still recorded
    span = _spans(exporter)[AGENT_SPAN_NAME]
    assert span.attributes is not None
    assert span.attributes[GEN_AI_COMPLETION] == "partial"


def test_agent_span_puts_agent_name_in_baggage_for_its_duration(tracer: Tracer) -> None:
    # GIVEN no agent name in baggage
    assert baggage.get_baggage(GEN_AI_AGENT_NAME_BAGGAGE_KEY) is None

    # WHEN inside agent_span
    with agent_span("researcher", tracer=tracer):
        # THEN outgoing requests would carry the agent name
        assert baggage.get_baggage(GEN_AI_AGENT_NAME_BAGGAGE_KEY) == "researcher"

    # THEN it does not leak past the span
    assert baggage.get_baggage(GEN_AI_AGENT_NAME_BAGGAGE_KEY) is None


def test_run_error_code_matches_core_agents() -> None:
    # GIVEN the duplicated constant (core.agents is not imported by agent_span)
    from datarobot_genai.core.agents import RUN_ERROR_CODE as CORE_RUN_ERROR_CODE

    # WHEN / THEN the two stay in sync
    assert RUN_ERROR_CODE == CORE_RUN_ERROR_CODE


def test_agent_span_imports_without_nat_or_core_extra() -> None:
    # GIVEN a fresh interpreter
    # WHEN only agent_span is imported
    code = (
        "import sys, datarobot_genai.core.telemetry.agent_span; "
        "heavy = [m for m in ('nat', 'datarobot_genai.core.agents', 'datarobot', 'openai') "
        "if m in sys.modules]; print(heavy)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    # THEN nothing from NAT or the core extra is loaded
    assert result.stdout.strip() == "[]"
