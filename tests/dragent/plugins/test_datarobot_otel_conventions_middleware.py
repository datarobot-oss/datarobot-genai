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

"""Unit tests for the DataRobot OTel conventions middleware.

The middleware wraps every workflow invocation in a ``datarobot_agent`` SDK
span and maps the last user message to ``gen_ai.prompt``, the workflow output to
``gen_ai.completion``, and tool-call starts to short-lived ``tool_name`` spans.
Tests drive the middleware against a real in-memory span exporter so assertions
look at the spans/attributes that would actually be exported.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from ag_ui.core import AssistantMessage
from ag_ui.core import RunAgentInput
from ag_ui.core import StepStartedEvent
from ag_ui.core import TextMessageChunkEvent
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import ToolCallStartEvent
from ag_ui.core import UserMessage
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import Message
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import datarobot_genai.dragent.plugins.datarobot_otel_conventions_middleware as mod
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.plugins.datarobot_otel_conventions_middleware import AGENT_SPAN_NAME
from datarobot_genai.dragent.plugins.datarobot_otel_conventions_middleware import GEN_AI_COMPLETION
from datarobot_genai.dragent.plugins.datarobot_otel_conventions_middleware import GEN_AI_PROMPT
from datarobot_genai.dragent.plugins.datarobot_otel_conventions_middleware import TOOL_NAME
from datarobot_genai.dragent.plugins.datarobot_otel_conventions_middleware import (
    DataRobotOtelConventionsMiddleware,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def span_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Point the module tracer at an in-memory exporter and isolate NAT bridging.

    ``SimpleSpanProcessor`` exports synchronously on span end, so finished spans
    are available immediately after each middleware call. The NAT context bridge
    is replaced with a no-op (it has its own tests) so spans are recorded under a
    plain local provider.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(mod, "tracer", provider.get_tracer(__name__))
    return exporter


@pytest.fixture
def middleware() -> DataRobotOtelConventionsMiddleware:
    return DataRobotOtelConventionsMiddleware(MagicMock(), MagicMock())


def _nat_input(content: str) -> ChatRequestOrMessage:
    return ChatRequestOrMessage(messages=[Message(role="user", content=content)])


def _nat_input_message(content: str) -> ChatRequestOrMessage:
    return ChatRequestOrMessage(input_message=content)


def _ag_ui_input(*messages: Any) -> RunAgentInput:
    return RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        messages=list(messages),
        tools=[],
        context=[],
        forwarded_props={},
        state={},
    )


def _call_next(result: Any):  # type: ignore[no-untyped-def]
    async def _inner(*args: Any, **kwargs: Any) -> Any:
        return result

    return _inner


def _call_next_stream(chunks: list[Any]):  # type: ignore[no-untyped-def]
    async def _inner(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        for chunk in chunks:
            yield chunk

    return _inner


def _span_named(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    spans = [span for span in exporter.get_finished_spans() if span.name == name]
    assert len(spans) == 1, f"expected exactly one {name!r} span, got {len(spans)}"
    return spans[0]


async def _drain(stream: AsyncIterator[Any]) -> list[Any]:
    return [chunk async for chunk in stream]


# ---------------------------------------------------------------------------
# Prompt extraction helpers
# ---------------------------------------------------------------------------


def test_last_user_message_content_nat_returns_last_user_message() -> None:
    request = ChatRequestOrMessage(
        messages=[
            Message(role="user", content="first"),
            Message(role="assistant", content="reply"),
            Message(role="user", content="last"),
        ]
    )
    assert mod._last_user_message_content(request) == "last"


def test_last_user_message_content_nat_skips_trailing_assistant() -> None:
    request = ChatRequestOrMessage(
        messages=[
            Message(role="user", content="question"),
            Message(role="assistant", content="answer"),
        ]
    )
    assert mod._last_user_message_content(request) == "question"


def test_last_user_message_content_nat_returns_none_without_user() -> None:
    request = ChatRequestOrMessage(messages=[Message(role="assistant", content="only assistant")])
    assert mod._last_user_message_content(request) is None


def test_last_user_message_content_nat_falls_back_to_input_message() -> None:
    # ChatRequestOrMessage may carry a bare string via input_message (no
    # messages list); the prompt should fall back to it.
    request = ChatRequestOrMessage(input_message="just a string prompt")
    assert mod._last_user_message_content(request) == "just a string prompt"


def test_last_user_message_content_nat_prefers_user_message_over_input_message() -> None:
    # input_message and messages are mutually exclusive on the model, but the
    # helper should still read user messages first when both are somehow present.
    request = ChatRequestOrMessage(messages=[Message(role="user", content="from messages")])
    request.input_message = "from input_message"
    assert mod._last_user_message_content(request) == "from messages"


def test_last_user_message_content_ag_ui_returns_last_user_message() -> None:
    run_input = _ag_ui_input(
        UserMessage(id="1", content="first"),
        AssistantMessage(id="2", role="assistant", content="reply"),
        UserMessage(id="3", content="last"),
    )
    assert mod._last_user_message_content(run_input) == "last"


def test_last_user_message_content_ag_ui_returns_none_without_user() -> None:
    run_input = _ag_ui_input(AssistantMessage(id="1", role="assistant", content="reply"))
    assert mod._last_user_message_content(run_input) is None


@pytest.mark.parametrize("value", ["a plain string", 123, None, {"messages": []}])
def test_last_user_message_content_unknown_type_returns_none(value: Any) -> None:
    assert mod._last_user_message_content(value) is None


def test_prompt_from_args_empty_returns_none() -> None:
    assert DataRobotOtelConventionsMiddleware._prompt_from_args(()) is None


# ---------------------------------------------------------------------------
# Completion extraction helpers
# ---------------------------------------------------------------------------


def test_response_text_joins_only_text_events() -> None:
    response = DRAgentEventResponse(
        events=[
            TextMessageContentEvent(message_id="m1", delta="Hello "),
            TextMessageChunkEvent(message_id="m1", delta="world"),
            StepStartedEvent(step_name="ignored"),
        ]
    )
    assert mod._response_text(response) == "Hello world"


def test_completion_from_output_str() -> None:
    assert DataRobotOtelConventionsMiddleware._completion_from_output("answer") == "answer"


def test_completion_from_output_event_response() -> None:
    response = DRAgentEventResponse(events=[TextMessageContentEvent(message_id="m1", delta="done")])
    assert DataRobotOtelConventionsMiddleware._completion_from_output(response) == "done"


@pytest.mark.parametrize("value", [123, None, ["x"]])
def test_completion_from_output_other_returns_none(value: Any) -> None:
    assert DataRobotOtelConventionsMiddleware._completion_from_output(value) is None


# ---------------------------------------------------------------------------
# function_middleware_invoke
# ---------------------------------------------------------------------------


async def test_invoke_sets_prompt_and_completion_for_str_output(
    middleware: DataRobotOtelConventionsMiddleware,
    span_exporter: InMemorySpanExporter,
) -> None:
    output = await middleware.function_middleware_invoke(
        _nat_input("What is 2+2?"),
        call_next=_call_next("4"),
        context=MagicMock(),
    )

    assert output == "4"
    span = _span_named(span_exporter, AGENT_SPAN_NAME)
    assert span.attributes[GEN_AI_PROMPT] == "What is 2+2?"
    assert span.attributes[GEN_AI_COMPLETION] == "4"


async def test_invoke_sets_prompt_from_input_message(
    middleware: DataRobotOtelConventionsMiddleware,
    span_exporter: InMemorySpanExporter,
) -> None:
    output = await middleware.function_middleware_invoke(
        _nat_input_message("bare string prompt"),
        call_next=_call_next("ok"),
        context=MagicMock(),
    )

    assert output == "ok"
    span = _span_named(span_exporter, AGENT_SPAN_NAME)
    assert span.attributes[GEN_AI_PROMPT] == "bare string prompt"


async def test_invoke_event_response_sets_completion_and_tool_spans(
    middleware: DataRobotOtelConventionsMiddleware,
    span_exporter: InMemorySpanExporter,
) -> None:
    response = DRAgentEventResponse(
        events=[
            ToolCallStartEvent(tool_call_id="t1", tool_call_name="lookup"),
            TextMessageContentEvent(message_id="m1", delta="done"),
        ]
    )

    output = await middleware.function_middleware_invoke(
        _ag_ui_input(UserMessage(id="1", content="hi")),
        call_next=_call_next(response),
        context=MagicMock(),
    )

    assert output is response
    agent_span = _span_named(span_exporter, AGENT_SPAN_NAME)
    assert agent_span.attributes[GEN_AI_PROMPT] == "hi"
    assert agent_span.attributes[GEN_AI_COMPLETION] == "done"

    tool_span = _span_named(span_exporter, "lookup")
    assert tool_span.attributes[TOOL_NAME] == "lookup"


async def test_invoke_unknown_input_sets_no_prompt(
    middleware: DataRobotOtelConventionsMiddleware,
    span_exporter: InMemorySpanExporter,
) -> None:
    output = await middleware.function_middleware_invoke(
        "unsupported-input-type",
        call_next=_call_next("answer"),
        context=MagicMock(),
    )

    assert output == "answer"
    span = _span_named(span_exporter, AGENT_SPAN_NAME)
    assert GEN_AI_PROMPT not in span.attributes
    assert span.attributes[GEN_AI_COMPLETION] == "answer"


async def test_invoke_non_text_output_sets_no_completion(
    middleware: DataRobotOtelConventionsMiddleware,
    span_exporter: InMemorySpanExporter,
) -> None:
    output = await middleware.function_middleware_invoke(
        _nat_input("hi"),
        call_next=_call_next(12345),
        context=MagicMock(),
    )

    assert output == 12345
    span = _span_named(span_exporter, AGENT_SPAN_NAME)
    assert GEN_AI_COMPLETION not in span.attributes


# ---------------------------------------------------------------------------
# function_middleware_stream
# ---------------------------------------------------------------------------


async def test_stream_aggregates_completion_across_chunks(
    middleware: DataRobotOtelConventionsMiddleware,
    span_exporter: InMemorySpanExporter,
) -> None:
    chunks = [
        DRAgentEventResponse(events=[TextMessageContentEvent(message_id="m1", delta="Hel")]),
        DRAgentEventResponse(events=[TextMessageContentEvent(message_id="m1", delta="lo")]),
    ]

    yielded = await _drain(
        middleware.function_middleware_stream(
            _nat_input("greet me"),
            call_next=_call_next_stream(chunks),
            context=MagicMock(),
        )
    )

    assert yielded == chunks
    span = _span_named(span_exporter, AGENT_SPAN_NAME)
    assert span.attributes[GEN_AI_PROMPT] == "greet me"
    assert span.attributes[GEN_AI_COMPLETION] == "Hello"


async def test_stream_sets_completion_when_closed_early_after_text(
    middleware: DataRobotOtelConventionsMiddleware,
    span_exporter: InMemorySpanExporter,
) -> None:
    """Completion survives early generator teardown after deltas were seen.

    Downstream moderation can finish and ``aclose()`` this generator (raising
    ``GeneratorExit`` at the ``yield``) before the source stream is exhausted.
    The ``gen_ai.completion`` attribute must still be attached from the deltas
    already accumulated — regression test for completion being set after the
    loop, which the teardown skipped.
    """
    chunks = [
        DRAgentEventResponse(events=[TextMessageContentEvent(message_id="m1", delta="Success")]),
        DRAgentEventResponse(events=[TextMessageContentEvent(message_id="m1", delta="-tail")]),
    ]

    stream = middleware.function_middleware_stream(
        _nat_input("go"),
        call_next=_call_next_stream(chunks),
        context=MagicMock(),
    )
    # Pull only the first (text) chunk, then close the generator early — mimics
    # moderation tearing it down before the ``async for`` exits normally.
    first = await stream.__anext__()
    assert isinstance(first, DRAgentEventResponse)
    await stream.aclose()

    span = _span_named(span_exporter, AGENT_SPAN_NAME)
    assert span.attributes[GEN_AI_PROMPT] == "go"
    assert span.attributes[GEN_AI_COMPLETION] == "Success"


async def test_stream_emits_tool_spans(
    middleware: DataRobotOtelConventionsMiddleware,
    span_exporter: InMemorySpanExporter,
) -> None:
    chunks = [
        DRAgentEventResponse(
            events=[ToolCallStartEvent(tool_call_id="t1", tool_call_name="search")]
        ),
        DRAgentEventResponse(events=[TextMessageContentEvent(message_id="m1", delta="result")]),
    ]

    await _drain(
        middleware.function_middleware_stream(
            _nat_input("find"),
            call_next=_call_next_stream(chunks),
            context=MagicMock(),
        )
    )

    tool_span = _span_named(span_exporter, "search")
    assert tool_span.attributes[TOOL_NAME] == "search"


async def test_stream_without_text_sets_no_completion(
    middleware: DataRobotOtelConventionsMiddleware,
    span_exporter: InMemorySpanExporter,
) -> None:
    chunks = [DRAgentEventResponse(events=[StepStartedEvent(step_name="step")])]

    await _drain(
        middleware.function_middleware_stream(
            _nat_input("hi"),
            call_next=_call_next_stream(chunks),
            context=MagicMock(),
        )
    )

    span = _span_named(span_exporter, AGENT_SPAN_NAME)
    assert GEN_AI_COMPLETION not in span.attributes


async def test_stream_passes_through_non_event_chunks(
    middleware: DataRobotOtelConventionsMiddleware,
    span_exporter: InMemorySpanExporter,
) -> None:
    chunks = ["raw-1", "raw-2"]

    yielded = await _drain(
        middleware.function_middleware_stream(
            "unsupported-input-type",
            call_next=_call_next_stream(chunks),
            context=MagicMock(),
        )
    )

    assert yielded == chunks
    span = _span_named(span_exporter, AGENT_SPAN_NAME)
    assert GEN_AI_PROMPT not in span.attributes
    assert GEN_AI_COMPLETION not in span.attributes
