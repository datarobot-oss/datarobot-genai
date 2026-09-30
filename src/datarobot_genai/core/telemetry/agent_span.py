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

"""DataRobot Tracing-table annotations for one agent invocation, without NAT.

The deployment Tracing table builds its Prompt / Completion / Tools columns from
span attributes (see
https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-tracing-code.html#map-spans-and-attributes-to-the-tracing-table),
and only from spans the entity emitted itself. Anything that runs an agent and
wants its own traces classified - the DRAgent NAT middleware, or an application
backend that calls an agent deployment over HTTP - wraps the call in
:func:`agent_span` and feeds the AG-UI events it sees to the returned recorder.

This module depends only on ``opentelemetry-api`` and ``ag-ui-protocol``, so it
is covered by the ``core`` extra and importable from a plain ``datarobot-genai``
install by callers that already have those two.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator
from contextlib import contextmanager

from ag_ui.core import BaseEvent
from ag_ui.core import EventType
from ag_ui.core import RunAgentInput
from ag_ui.core import RunErrorEvent
from ag_ui.core import ToolCallStartEvent
from opentelemetry import baggage
from opentelemetry import trace
from opentelemetry.trace import Span
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode
from opentelemetry.trace import Tracer

from datarobot_genai.core.telemetry.agent_identity import GEN_AI_AGENT_NAME_BAGGAGE_KEY
from datarobot_genai.core.telemetry.agent_identity import agent_name_baggage

_tracer = trace.get_tracer(__name__)

# Parent span created per invocation so the Tracing table attributes always
# have a span to live on.
AGENT_SPAN_NAME = "datarobot_agent"

# Span attributes that map to deployment Tracing table columns. Spelled out
# rather than taken from ``datarobot_opentelemetry.semconv.SpanAttributes``
# (same values) so this module needs nothing beyond the ``core`` extra.
GEN_AI_PROMPT = "gen_ai.prompt"  # Prompt column
GEN_AI_COMPLETION = "gen_ai.completion"  # Completion column
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"  # Tools column
DATAROBOT_SESSION_ID = "datarobot.session_id"
ERROR_TYPE = "error.type"  # Failed span classification

# Same value as ``datarobot_genai.core.agents.RUN_ERROR_CODE``; not imported from
# there because that package pulls in the ``core`` extra.
RUN_ERROR_CODE = "RUN_ERROR"

# AG-UI event types that carry assistant text deltas.
_TEXT_EVENT_TYPES = (EventType.TEXT_MESSAGE_CONTENT, EventType.TEXT_MESSAGE_CHUNK)


def last_user_message(run_agent_input: RunAgentInput) -> str | None:
    """Return the content of the last ``user`` message in an AG-UI input.

    Matches on ``role`` rather than ``isinstance(UserMessage)``: callers that
    rebuild ``messages`` from stored chat history (for example the agent
    application's backend) hold generic message objects, not ``UserMessage``.
    """
    for message in reversed(run_agent_input.messages):
        if getattr(message, "role", None) == "user":
            content = getattr(message, "content", None)
            return None if content is None else str(content)
    return None


class AgentSpanRecorder:
    """Writes what an agent streams back onto its :func:`agent_span`.

    Feed it every AG-UI event the agent produces, in order, via :meth:`observe`.
    Text deltas are aggregated into ``gen_ai.completion`` when the span closes,
    each ``ToolCallStart`` becomes a short-lived child span carrying
    ``gen_ai.tool.name`` (Tools column), and a ``RunError`` marks the span failed.
    """

    def __init__(self, span: Span, tracer: Tracer) -> None:
        self._span = span
        self._tracer = tracer
        self._parts: list[str] = []
        self._completion: str | None = None

    @property
    def span(self) -> Span:
        return self._span

    def observe(self, events: Iterable[BaseEvent]) -> None:
        for event in events:
            if event.type in _TEXT_EVENT_TYPES:
                delta = getattr(event, "delta", None)
                if delta:
                    self._parts.append(delta)
            elif isinstance(event, ToolCallStartEvent):
                self._emit_tool_call_span(event)
            elif isinstance(event, RunErrorEvent):
                self._mark_error(event)

    def set_completion(self, completion: str) -> None:
        """Record the completion directly, overriding any aggregated text deltas.

        For callers whose output is not a stream of AG-UI events (e.g. NAT's
        non-streaming path, which returns a plain ``str``).
        """
        self._completion = completion

    def finish(self) -> None:
        """Write the completion onto the span. Called by :func:`agent_span` on exit."""
        completion = self._completion
        if completion is None and self._parts:
            completion = "".join(self._parts)
        if completion is not None:
            self._span.set_attribute(GEN_AI_COMPLETION, completion)

    def _emit_tool_call_span(self, event: ToolCallStartEvent) -> None:
        # Tool execution usually happens where we can't wrap it (inside NAT, or
        # in a remote agent), but its ToolCallStart event is visible. A span that
        # is created and immediately ended with ``gen_ai.tool.name`` is enough to
        # populate the Tools column. ``gen_ai.agent.name`` is stamped on the same
        # span (not left only in baggage) because Datavolt's agent/tool cross-tab
        # needs both attributes on one span.
        with self._tracer.start_as_current_span(event.tool_call_name) as span:
            span.set_attribute(GEN_AI_TOOL_NAME, event.tool_call_name)
            agent_name = baggage.get_baggage(GEN_AI_AGENT_NAME_BAGGAGE_KEY)
            if agent_name:
                span.set_attribute(GEN_AI_AGENT_NAME, str(agent_name))

    def _mark_error(self, event: RunErrorEvent) -> None:
        self._span.set_attribute(ERROR_TYPE, event.code or RUN_ERROR_CODE)
        self._span.set_status(Status(StatusCode.ERROR, event.message or "workflow run error"))


@contextmanager
def agent_span(
    agent_name: str,
    *,
    prompt: str | None = None,
    session_id: str | None = None,
    tracer: Tracer | None = None,
) -> Iterator[AgentSpanRecorder]:
    """Wrap one agent invocation in a ``datarobot_agent`` span.

    Sets ``gen_ai.agent.name`` (and puts it in baggage for the duration, so tool
    spans and outgoing requests carry it), ``gen_ai.prompt`` and
    ``datarobot.session_id`` when given, and writes ``gen_ai.completion`` on exit
    - also when the caller's stream is closed early, since the span outlives the
    recorder's ``finish``.

    Pass ``tracer`` to record under a specific tracer (callers keep their own
    module-level tracer so tests can point it at an in-memory exporter).
    """
    active_tracer = tracer or _tracer
    with (
        active_tracer.start_as_current_span(AGENT_SPAN_NAME) as span,
        agent_name_baggage(agent_name),
    ):
        span.set_attribute(GEN_AI_AGENT_NAME, agent_name)
        if prompt is not None:
            span.set_attribute(GEN_AI_PROMPT, prompt)
        if session_id is not None:
            span.set_attribute(DATAROBOT_SESSION_ID, session_id)
        recorder = AgentSpanRecorder(span, active_tracer)
        try:
            yield recorder
        finally:
            recorder.finish()
