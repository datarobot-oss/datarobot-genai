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

"""In-process mock OTLP/HTTP collector and tracing assertions for e2e tests.

The OTel SDK HTTP exporter and NAT's ``OTLPSpanAdapterExporter`` both POST
spans via ``requests.Session()``; an in-process ``responses``/``respx`` patch
in the test process would not intercept the dragent *subprocess*. A real
listener on ``127.0.0.1:<port>`` does.

The dragent server is started in a separate process *before* pytest, so the
collector is bound to a *fixed* port (see ``conftest.otel_collector``) that the
server's ``OTEL_EXPORTER_OTLP_ENDPOINT`` already points at. The server buffers
spans and exports them on its flush schedule, so spans generated while a test
runs (after the collector is up) are captured even though the server outlived
the collector's startup.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from typing import Any

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

# OTLP/HTTP traces ingest path. The DataRobot collector ingress lives off
# ``/otel/v1/traces`` (see ``resolve_otel_traces_endpoint_from_env``).
OTLP_TRACES_PATH = "/otel/v1/traces"

# --- OpenTelemetry tracing config ------------------------------------------
# The dragent server (started before pytest) is configured to export spans to
# the in-process mock collector via OTEL_EXPORTER_OTLP_ENDPOINT /
# OTEL_EXPORTER_OTLP_HEADERS. These constants MUST match the values the server
# is launched with in ``e2e-tests/dragent/Taskfile.yaml``.
MOCK_OTEL_COLLECTOR_PORT = int(os.environ.get("MOCK_OTEL_COLLECTOR_PORT", "4318"))
# OTLP base endpoint the server exports to; ``/v1/traces`` is appended by the
# DataRobot exporter (see resolve_otel_traces_endpoint_from_env).
OTEL_EXPORTER_OTLP_ENDPOINT = f"http://localhost:{MOCK_OTEL_COLLECTOR_PORT}/otel"
# Sentinel DataRobot auth headers; the mock collector does not validate them,
# the tests only assert they reached the ingest unmodified.
OTEL_API_KEY = "e2e-otel-token"
OTEL_ENTITY_ID = "deployment-e2e-test"
OTEL_EXPORTER_OTLP_HEADERS = (
    f"X-DataRobot-Api-Key={OTEL_API_KEY},X-DataRobot-Entity-Id={OTEL_ENTITY_ID}"
)

# Span attributes that map to the deployment Tracing table columns, per
# https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-tracing-code.html#map-spans-and-attributes-to-the-tracing-table
# Mirrors the constants in
# ``datarobot_genai.dragent.plugins.datarobot_otel_conventions_middleware``.
GEN_AI_PROMPT = "gen_ai.prompt"  # Prompt column
GEN_AI_COMPLETION = "gen_ai.completion"  # Completion column
TOOL_NAME = "tool_name"  # Tools column


@dataclass(frozen=True)
class CapturedRequest:
    path: str
    headers: dict[str, str]
    body: bytes


@dataclass(frozen=True)
class ExportedSpan:
    """A single span parsed out of an OTLP/HTTP export body."""

    name: str
    trace_id: bytes
    span_id: bytes
    parent_span_id: bytes
    attributes: dict[str, Any]


def _anyvalue_to_python(value: Any) -> Any:
    """Unwrap an OTLP ``AnyValue`` into a plain Python scalar (best effort)."""
    if value.HasField("string_value"):
        return value.string_value
    if value.HasField("bool_value"):
        return value.bool_value
    if value.HasField("int_value"):
        return value.int_value
    if value.HasField("double_value"):
        return value.double_value
    # array/kvlist/bytes are not asserted on by the tracing tests; surface a
    # stable repr so a mismatch is still debuggable.
    return None


@dataclass
class _State:
    requests: list[CapturedRequest] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class MockOtelCollector:
    """Listens on 127.0.0.1:<port>; records every POST and exposes parsed spans."""

    def __init__(self, port: int = 0) -> None:
        self._state = _State()
        # ``port=0`` lets the kernel pick a free port (handy for standalone
        # use); the e2e fixture pins a fixed port the dragent server targets.
        self._server = ThreadingHTTPServer(("127.0.0.1", port), self._build_handler())
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="mock-otel-collector",
            daemon=True,
        )

    @property
    def requests(self) -> list[CapturedRequest]:
        with self._state.lock:
            return list(self._state.requests)

    @property
    def endpoint(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def reset(self) -> None:
        """Drop captured requests (e.g. between tests that assert in isolation)."""
        with self._state.lock:
            self._state.requests.clear()

    def spans(self, path: str = OTLP_TRACES_PATH) -> list[ExportedSpan]:
        """Parse every captured OTLP body at *path* into flattened spans."""
        out: list[ExportedSpan] = []
        for request in self.requests:
            if request.path != path or not request.body:
                continue
            message = ExportTraceServiceRequest()
            try:
                message.ParseFromString(request.body)
            except Exception:  # noqa: BLE001 — skip a malformed/partial body
                continue
            for resource_spans in message.resource_spans:
                for scope_spans in resource_spans.scope_spans:
                    for span in scope_spans.spans:
                        out.append(
                            ExportedSpan(
                                name=span.name,
                                trace_id=span.trace_id,
                                span_id=span.span_id,
                                parent_span_id=span.parent_span_id,
                                attributes={
                                    attribute.key: _anyvalue_to_python(attribute.value)
                                    for attribute in span.attributes
                                },
                            )
                        )
        return out

    def wait_for_requests(self, n: int = 1, timeout: float = 10.0) -> None:
        """Block until at least *n* requests have been captured, or raise."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._state.lock:
                if len(self._state.requests) >= n:
                    return
            time.sleep(0.05)
        raise AssertionError(
            f"Timed out waiting for {n} request(s); captured {len(self.requests)}."
        )

    def wait_for_spans(
        self,
        predicate: Callable[[ExportedSpan], bool],
        *,
        timeout: float = 30.0,
        poll: float = 0.25,
    ) -> list[ExportedSpan]:
        """Poll until ≥ 1 exported span matches *predicate*; return all matches.

        Returns an empty list on timeout so callers can build a rich assertion
        message from the spans that *were* observed.
        """
        deadline = time.monotonic() + timeout
        while True:
            matched = [span for span in self.spans() if predicate(span)]
            if matched or time.monotonic() >= deadline:
                return matched
            time.sleep(poll)

    def __enter__(self) -> MockOtelCollector:
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        state = self._state

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — http.server API
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b""
                with state.lock:
                    state.requests.append(
                        CapturedRequest(
                            path=self.path,
                            headers={k: v for k, v in self.headers.items()},
                            body=body,
                        )
                    )
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802 — http.server API
                # The moderation middleware (datarobot_dome) probes
                # ``<DATAROBOT_ENDPOINT>/account/info/`` at builder time to
                # validate the API token. Return an empty 200 so the workflow
                # can start; we don't otherwise assert on GETs.
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, format: str, *args: object) -> None:
                # Quiet — the test asserts on captured state, not stderr noise.
                return

        return _Handler


def _assert_datarobot_auth_headers(collector: MockOtelCollector) -> None:
    """At least one trace export must carry the DataRobot ingest auth headers.

    HTTP header names are case-insensitive, so compare on lowercased keys
    rather than relying on the exporter's exact casing.
    """

    def lower(headers: dict[str, str]) -> dict[str, str]:
        return {k.lower(): v for k, v in headers.items()}

    authed = [
        request
        for request in collector.requests
        if request.path == OTLP_TRACES_PATH
        and lower(request.headers).get("x-datarobot-api-key") == OTEL_API_KEY
        and lower(request.headers).get("x-datarobot-entity-id") == OTEL_ENTITY_ID
    ]
    assert authed, (
        f"Expected ≥ 1 POST to {OTLP_TRACES_PATH} with DR auth headers "
        f"(X-DataRobot-Api-Key={OTEL_API_KEY}, X-DataRobot-Entity-Id={OTEL_ENTITY_ID}); "
        f"captured {len(collector.requests)} request(s) at paths "
        f"{sorted({r.path for r in collector.requests})} with header keys "
        f"{[sorted(r.headers.keys()) for r in collector.requests if r.path == OTLP_TRACES_PATH]}."
    )


def _describe_span(span: ExportedSpan) -> str:
    """One-line, debuggable description of a span (name, ids, attributes).

    Attributes are the key signal for tracking down an orphaned span (e.g. the
    ``http.*`` / ``url.*`` attributes pin down which outbound call escaped the
    agent trace), so they are always included.
    """
    if span.attributes:
        attrs = ", ".join(f"{k}={v!r}" for k, v in sorted(span.attributes.items()))
    else:
        attrs = "<none>"
    return (
        f"name={span.name!r} span_id={span.span_id.hex()} "
        f"trace_id={span.trace_id.hex()} attrs={{{attrs}}}"
    )


# OTel semantic-convention attribute keys that carry a span's request URL. The
# requests/httpx instrumentors use ``http.url`` (legacy) or ``url.full`` (stable
# semconv); check both so the URL-ignore filter is convention-agnostic.
_SPAN_URL_ATTRIBUTE_KEYS = ("http.url", "url.full")


def _span_url(span: ExportedSpan) -> str | None:
    for key in _SPAN_URL_ATTRIBUTE_KEYS:
        value = span.attributes.get(key)
        if isinstance(value, str):
            return value
    return None


def _span_url_is_ignored(span: ExportedSpan, ignore_span_urls: Sequence[str]) -> bool:
    if not ignore_span_urls:
        return False
    url = _span_url(span)
    if url is None:
        return False
    return any(fragment in url for fragment in ignore_span_urls)


def _assert_single_trace_id(
    collector: MockOtelCollector,
    *,
    agent_trace_ids: set[bytes],
    ignore_span_urls: Sequence[str] = (),
) -> None:
    """Every span exported for the current run must share one ``trace_id``.

    A single agent invocation must produce a single, connected trace: the
    per-invocation ``datarobot_agent`` root span, its framework children
    (workflow, guards, tool-call spans), *and* every outbound HTTP call made
    while serving the request must share one ``trace_id``. A break in OTel
    context propagation surfaces here as spans split across multiple traces —
    commonly bootstrap/setup HTTP client spans (token validation, guard/MCP
    setup) that were started without the agent span as their parent.

    *agent_trace_ids* is the trace_id set of the matched ``datarobot_agent``
    span(s); it identifies the expected trace so orphans can be contrasted
    against it in the failure output.

    *ignore_span_urls* holds URL fragments (substring match against a span's
    ``http.url`` / ``url.full``) for HTTP spans that legitimately root their own
    trace because they fire at *import* time — before any workflow trace exists
    (e.g. LiteLLM's model-cost-map fetch, the DataRobot client version check).
    Such spans are dropped before the single-trace check.

    Relies on the ``_reset_otel_collector`` autouse fixture clearing the
    session-scoped collector before each test, so the captured spans belong to
    the run under test only.
    """
    all_spans = [
        span
        for span in collector.spans()
        if not _span_url_is_ignored(span, ignore_span_urls)
    ]
    trace_ids = {span.trace_id for span in all_spans}
    if len(trace_ids) == 1:
        return

    # The agent invocation's trace is the reference; anything outside it is an
    # orphan. Fall back to the most populated trace if the agent span's trace is
    # ambiguous (0 or >1 distinct ids).
    if len(agent_trace_ids) == 1:
        main_trace = next(iter(agent_trace_ids))
    else:
        main_trace = max(trace_ids, key=lambda tid: sum(s.trace_id == tid for s in all_spans))

    spans_by_trace: dict[bytes, list[ExportedSpan]] = {}
    for span in all_spans:
        spans_by_trace.setdefault(span.trace_id, []).append(span)

    orphans = [span for span in all_spans if span.trace_id != main_trace]

    main_names = sorted(span.name for span in spans_by_trace.get(main_trace, []))
    orphan_traces = sorted(
        (tid.hex(), sorted(s.name for s in spans))
        for tid, spans in spans_by_trace.items()
        if tid != main_trace
    )
    orphan_details = "\n".join(f"    - {_describe_span(span)}" for span in orphans)

    raise AssertionError(
        f"Expected all {len(all_spans)} exported span(s) to share one trace_id, "
        f"but found {len(trace_ids)} distinct trace_id(s).\n"
        f"  Agent trace {main_trace.hex()} ({len(main_names)} span(s)): {main_names}\n"
        f"  Orphan trace(s) ({len(orphans)} span(s) across {len(orphan_traces)} trace(s)): "
        f"{orphan_traces}\n"
        f"  Orphaned span details:\n{orphan_details}"
    )


def _assert_crewai_spans(collector: MockOtelCollector, agent_trace_ids: set[bytes]) -> None:
    """Assert CrewAI auto-instrumentor spans are present in the agent's trace.

    The CrewAI instrumentor (both the synchronous ``kickoff`` path and the async
    ``akickoff`` path wired up by
    ``datarobot_genai.crewai.telemetry.DataRobotCrewAIInstrumentor``) emits a
    ``crewai.workflow`` root span, one ``<agent role>.agent`` span per agent,
    one ``<task description>.task`` span per task, and one ``<model>.llm`` span
    per LLM call. This guards against the framework instrumentation silently
    going away — those spans would vanish even though ``gen_ai.prompt`` /
    ``gen_ai.completion`` (set by the middleware, not the framework) still appear.
    """
    names = [span.name for span in collector.spans() if span.trace_id in agent_trace_ids]
    missing = [
        label
        for label, present in (
            ("crewai.workflow", any(n == "crewai.workflow" for n in names)),
            ("<agent>.agent", any(n.endswith(".agent") for n in names)),
            ("<task>.task", any(n.endswith(".task") for n in names)),
            ("<model>.llm", any(n.endswith(".llm") for n in names)),
        )
        if not present
    ]
    assert not missing, (
        f"Expected CrewAI span(s) {missing} in the agent trace, "
        f"but observed span names {sorted(set(names))}."
    )


def _is_nat_internal_span(span: ExportedSpan) -> bool:
    """Whether *span* was emitted by NAT's own function-tracing pipeline.

    NAT tags the spans it creates with ``nat.*`` attributes (e.g.
    ``nat.span.kind``, ``nat.function.name``). Those are excluded from the
    duplicate-span check below because NAT's workflow function layer legitimately
    self-nests a ``<workflow>`` FUNCTION span under another ``<workflow>`` span —
    an artifact of NAT's tracing, not of the instrumentation this repo owns.
    Framework auto-instrumentor spans (``crewai.*``) and the custom
    ``{model}.llm`` spans never carry ``nat.*`` attributes, so the check still
    covers everything we actually control.
    """
    return any(key.startswith("nat.") for key in span.attributes)


# Traceloop ``span.kind`` values for the grouping "entity" spans that wrap a
# whole workflow or a single traced method, rather than one leaf operation.
_TRACELOOP_ENTITY_SPAN_KINDS = frozenset({"workflow", "task"})


def _is_traceloop_entity_span(span: ExportedSpan) -> bool:
    """Whether *span* is a Traceloop workflow/task grouping span.

    Traceloop-based instrumentors (e.g. llamaindex) tag each traced entity span
    with ``traceloop.span.kind`` and name it after the *class/method*, not the
    operation. When an agent recursively invokes the same entity (e.g.
    llamaindex's ``DataRobotLiteLLM.task`` calling itself for a follow-up chat
    completion), that legitimately nests a ``<Entity>.task`` span directly under
    another ``<Entity>.task`` span of the same name. That is expected recursion,
    not double-counted instrumentation, so these spans are excluded from the
    duplicate-span check below.

    The leaf operations this repo actually guards against double-wrapping — the
    custom ``{model}.llm`` CLIENT spans and the CrewAI auto-instrumentor spans —
    never carry ``traceloop.span.kind``, so the check still covers them.
    """
    return span.attributes.get("traceloop.span.kind") in _TRACELOOP_ENTITY_SPAN_KINDS


def _assert_no_duplicate_spans(
    collector: MockOtelCollector, agent_trace_ids: set[bytes]
) -> None:
    """No span may be a direct child of another span with the same name.

    Duplicate instrumentation double-counts a single logical operation as two
    identically-named spans nested one inside the other — e.g. ``crewai.workflow``
    under ``crewai.workflow``, ``<agent>.agent`` under ``<agent>.agent``,
    ``<task>.task`` under ``<task>.task``, or ``{model}.llm`` under ``{model}.llm``.

    The ``{model}.llm`` case is the concrete risk: datarobot-genai's custom
    ``crewai.llm.LLM`` subclasses (``LitellmStopWordLLM``, ``RouterLitellmOnlyLLM``)
    emit an LLM span for the native tool-calling branch (which never delegates to
    ``super().call()/acall()``), while the CrewAI LLM instrumentation emits one
    for the non-native branch (reached via that ``super()`` call). The branches
    are mutually exclusive, so each call must yield exactly one span. The check is
    deliberately name-agnostic and framework-agnostic so it guards every agent's
    spans against any future double-wrapping.

    Spans that some frameworks legitimately self-nest are excluded: NAT-internal
    spans (see :func:`_is_nat_internal_span`) and Traceloop workflow/task entity
    spans (see :func:`_is_traceloop_entity_span`). Both are grouping spans named
    after a function/entity that can recurse, not double-counted instrumentation
    of a single operation.
    """
    spans = [
        span
        for span in collector.spans()
        if span.trace_id in agent_trace_ids
        and not _is_nat_internal_span(span)
        and not _is_traceloop_entity_span(span)
    ]
    # Parents may be any captured span in the trace (a duplicated child's parent
    # is itself, by name), so index every span, then flag non-NAT children whose
    # parent shares their name.
    by_span_id = {span.span_id: span for span in collector.spans()}
    duplicates = [
        span
        for span in spans
        if span.parent_span_id in by_span_id
        and by_span_id[span.parent_span_id].name == span.name
    ]
    assert not duplicates, (
        "Found span(s) nested directly under a same-named parent span, i.e. "
        "duplicate/double-counted instrumentation for a single operation:\n"
        + "\n".join(f"    - {_describe_span(span)}" for span in duplicates)
    )


# Per-framework span assertions, keyed by framework/AGENT name (see
# ``helpers.AGENT``). Add a ``_assert_<framework>_spans`` function and register
# it here to cover another framework. A framework absent from this map
# contributes no extra assertions.
_FRAMEWORK_SPAN_ASSERTERS: dict[str, Callable[[MockOtelCollector, set[bytes]], None]] = {
    "crewai": _assert_crewai_spans,
}


def assert_tracing_conventions(
    collector: MockOtelCollector,
    prompt: str,
    *,
    expect_tool_name: bool = False,
    framework: str | None = None,
    ignore_span_urls: Sequence[str] = (),
    timeout: float = 30.0,
) -> None:
    """Assert the agent exported DataRobot Tracing-table spans for *prompt*.

    Finds this request's spans by ``gen_ai.prompt`` (the verbatim user message
    recorded by the ``datarobot_otel_conventions`` middleware), then asserts the
    convention attributes are present:

    * ``gen_ai.prompt`` and ``gen_ai.completion`` on the per-invocation
      ``datarobot_agent`` span,
    * every exported span for the run (including outbound HTTP calls) shares a
      single ``trace_id``, and
    * the export carried the DataRobot ingest auth headers.

    When *expect_tool_name* is set, also requires a ``tool_name`` span in the
    same trace (frameworks that surface tool calls as AG-UI events).

    When *framework* is set (e.g. ``"crewai"``), also requires that framework's
    characteristic auto-instrumentor spans in the same trace — see
    ``_FRAMEWORK_SPAN_ASSERTERS``. Unknown frameworks add no assertions.

    *ignore_span_urls* excludes import-time HTTP client spans (matched by URL
    fragment) from the single-trace check — see ``_assert_single_trace_id``.
    """
    agent_spans = collector.wait_for_spans(
        lambda span: span.attributes.get(GEN_AI_PROMPT) == prompt
        and GEN_AI_COMPLETION in span.attributes,
        timeout=timeout,
    )
    if not agent_spans:
        observed = sorted(
            str(s.attributes[GEN_AI_PROMPT])
            for s in collector.spans()
            if GEN_AI_PROMPT in s.attributes
        )
        raise AssertionError(
            f"No exported span carried {GEN_AI_PROMPT}=={prompt!r} together with "
            f"{GEN_AI_COMPLETION}. Observed prompts: {observed}"
        )

    agent_trace_ids = {span.trace_id for span in agent_spans}

    _assert_single_trace_id(
        collector,
        agent_trace_ids=agent_trace_ids,
        ignore_span_urls=ignore_span_urls,
    )

    # Runs for every agent/framework: no operation may be double-counted as two
    # same-named nested spans (see _assert_no_duplicate_spans).
    _assert_no_duplicate_spans(collector, agent_trace_ids)

    _assert_datarobot_auth_headers(collector)

    if framework and (assert_framework_spans := _FRAMEWORK_SPAN_ASSERTERS.get(framework)):
        assert_framework_spans(collector, agent_trace_ids)

    if expect_tool_name:
        spans_in_trace = [s for s in collector.spans() if s.trace_id in agent_trace_ids]
        tool_spans = [s for s in spans_in_trace if s.attributes.get(TOOL_NAME)]
        assert tool_spans, (
            f"Expected a span carrying {TOOL_NAME!r} in the same trace as the "
            f"agent span for the tool-calling request; "
            f"got span names {sorted({s.name for s in spans_in_trace})}."
        )
