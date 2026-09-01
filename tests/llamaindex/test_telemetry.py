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

from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from datarobot_opentelemetry.semconv import SpanAttributes as DataRobotSpanAttributes
from opentelemetry import baggage
from opentelemetry.sdk.trace import Span
from opentelemetry.sdk.trace import TracerProvider

from datarobot_genai.llama_index import telemetry


@pytest.fixture(autouse=True)
def reset_state() -> Iterator[None]:
    """Reset the module-level instrumentation flag around each test."""
    telemetry._INSTRUMENTED["llamaindex"] = False
    yield
    telemetry._INSTRUMENTED["llamaindex"] = False


def test_instrument_enables_instrumentor() -> None:
    with (
        patch.object(telemetry, "LlamaIndexInstrumentor") as instrumentor,
        patch.object(telemetry, "get_dispatcher"),
    ):
        telemetry.instrument()

    instrumentor.return_value.instrument.assert_called_once()
    assert telemetry._INSTRUMENTED["llamaindex"] is True


def test_instrument_is_idempotent() -> None:
    with (
        patch.object(telemetry, "LlamaIndexInstrumentor") as instrumentor,
        patch.object(telemetry, "get_dispatcher") as get_dispatcher,
    ):
        telemetry.instrument()
        telemetry.instrument()

    instrumentor.return_value.instrument.assert_called_once()
    get_dispatcher.return_value.add_span_handler.assert_called_once()


def test_instrument_swallows_errors() -> None:
    with patch.object(telemetry, "LlamaIndexInstrumentor") as instrumentor:
        instrumentor.return_value.instrument.side_effect = RuntimeError("boom")
        # Should not raise
        telemetry.instrument()

    assert telemetry._INSTRUMENTED["llamaindex"] is False


def test_instrument_registers_agent_name_span_handler() -> None:
    with (
        patch.object(telemetry, "LlamaIndexInstrumentor"),
        patch.object(telemetry, "get_dispatcher") as get_dispatcher,
    ):
        telemetry.instrument()

    (registered_handler,), _ = get_dispatcher.return_value.add_span_handler.call_args
    assert isinstance(registered_handler, telemetry._AgentNameSpanHandler)


# ---------------------------------------------------------------------------
# _AgentNameSpanHandler
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_span() -> Iterator[Span]:
    """Return a real, currently-active (not yet ended) span - mirrors what
    OpenLLMetry's own span handler would have already opened and attached to
    context by the time ours runs (it's registered second). Still-open SDK
    spans expose `.attributes` directly, so assertions don't need to wait for
    export.
    """
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("FunctionAgent.task") as span:
        yield cast(Span, span)


def test_span_enter_sets_agent_name_when_event_carries_it(recording_span: Span) -> None:
    handler = telemetry._AgentNameSpanHandler()
    ev = SimpleNamespace(current_agent_name="researcher")
    bound_args = MagicMock(arguments={"ev": ev})

    handler.span_enter(id_="FunctionAgent.run_agent_step-1", bound_args=bound_args)

    assert recording_span.attributes is not None
    assert recording_span.attributes[DataRobotSpanAttributes.GEN_AI_AGENT_NAME] == "researcher"


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({}, id="no-ev-argument"),
        pytest.param({"ev": SimpleNamespace()}, id="ev-without-current-agent-name"),
        pytest.param(
            {"ev": SimpleNamespace(current_agent_name=None)}, id="current-agent-name-none"
        ),
    ],
)
def test_span_enter_is_a_noop_without_a_usable_agent_name(
    recording_span: Span,
    arguments: dict[str, object],
) -> None:
    handler = telemetry._AgentNameSpanHandler()
    bound_args = MagicMock(arguments=arguments)

    handler.span_enter(id_="SomeOtherStep-1", bound_args=bound_args)

    assert recording_span.attributes is not None
    assert DataRobotSpanAttributes.GEN_AI_AGENT_NAME not in recording_span.attributes


def test_new_span_and_span_lifecycle_hooks_are_noops() -> None:
    """These exist only to satisfy BaseSpanHandler's abstract interface -
    all the real logic is in span_enter above.
    """
    handler = telemetry._AgentNameSpanHandler()
    bound_args = MagicMock(arguments={})

    handler.new_span(id_="x", bound_args=bound_args)
    handler.prepare_to_exit_span(id_="x", bound_args=bound_args)
    handler.prepare_to_drop_span(id_="x", bound_args=bound_args)


@pytest.mark.parametrize("outcome", ["exit", "drop"])
def test_full_dispatcher_lifecycle_does_not_raise(outcome: str) -> None:
    """span_enter is overridden directly and never populates `open_spans`
    (unlike the base class's own span_enter, which populates it from
    new_span's return value). The dispatcher still calls the inherited
    span_exit/span_drop on every handler regardless - those delete from
    open_spans only when prepare_to_exit_span/prepare_to_drop_span return a
    truthy span, so this only holds together because both are stubbed to
    return None. Exercises the real dispatcher-facing methods, not just the
    prepare_to_* stubs directly, since that's the coupling that matters.
    """
    handler = telemetry._AgentNameSpanHandler()
    bound_args = MagicMock(arguments={})

    handler.span_enter(id_="x", bound_args=bound_args, instance=None, parent_id=None, tags=None)
    if outcome == "exit":
        handler.span_exit(id_="x", bound_args=bound_args, instance=None, result=None)
    else:
        handler.span_drop(id_="x", bound_args=bound_args, instance=None, err=RuntimeError("boom"))

    assert handler.open_spans == {}


# ---------------------------------------------------------------------------
# Agent name propagated as Baggage for the duration of the span
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["exit", "drop"])
def test_span_enter_attaches_baggage_and_span_exit_or_drop_detaches_it(
    recording_span: Span, outcome: str
) -> None:
    handler = telemetry._AgentNameSpanHandler()
    ev = SimpleNamespace(current_agent_name="researcher")
    bound_args = MagicMock(arguments={"ev": ev})

    handler.span_enter(id_="step-1", bound_args=bound_args)
    assert baggage.get_baggage("gen_ai.agent.name") == "researcher"

    if outcome == "exit":
        handler.span_exit(id_="step-1", bound_args=bound_args, result=None)
    else:
        handler.span_drop(id_="step-1", bound_args=bound_args, err=RuntimeError("boom"))

    assert baggage.get_baggage("gen_ai.agent.name") is None
    assert handler._baggage_tokens == {}


def test_span_enter_without_a_usable_agent_name_attaches_no_baggage(recording_span: Span) -> None:
    handler = telemetry._AgentNameSpanHandler()
    bound_args = MagicMock(arguments={})

    handler.span_enter(id_="step-1", bound_args=bound_args)

    assert baggage.get_baggage("gen_ai.agent.name") is None
    assert handler._baggage_tokens == {}


def test_nested_spans_each_detach_their_own_baggage_independently(recording_span: Span) -> None:
    """Baggage attach/detach must layer like a stack: the inner span's detach
    must not clobber the outer span's still-active baggage.
    """
    handler = telemetry._AgentNameSpanHandler()
    outer_ev = SimpleNamespace(current_agent_name="outer_agent")
    inner_ev = SimpleNamespace(current_agent_name="inner_agent")

    handler.span_enter(id_="outer", bound_args=MagicMock(arguments={"ev": outer_ev}))
    assert baggage.get_baggage("gen_ai.agent.name") == "outer_agent"

    handler.span_enter(id_="inner", bound_args=MagicMock(arguments={"ev": inner_ev}))
    assert baggage.get_baggage("gen_ai.agent.name") == "inner_agent"

    handler.span_exit(id_="inner", bound_args=MagicMock(arguments={}), result=None)
    assert baggage.get_baggage("gen_ai.agent.name") == "outer_agent"

    handler.span_exit(id_="outer", bound_args=MagicMock(arguments={}), result=None)
    assert baggage.get_baggage("gen_ai.agent.name") is None
