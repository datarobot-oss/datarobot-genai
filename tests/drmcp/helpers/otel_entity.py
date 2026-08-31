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

"""Self-provisioned OTel entity for the tier-3 OTel acceptance cases.

The acceptance cases in ``tests/drmcp/acceptance/test_otel_tools.py`` need an entity
that carries OTel data with a known shape: at least one error-status trace whose
failing span holds LLM output, and at least one error-level log line. Pointing them
at whatever real entity happens to exist is not reproducible -- its traces change,
get deleted, or belong to a cluster the next developer cannot reach.

So the fixture builds its own. The entity is a **Use Case** (OTel entity type
``experiment_container``), which is the same target the ``dr xp`` experimentation
dashboard and the ``datarobot-external-agent-monitoring`` skill use for
externally-instrumented agents: creating one needs nothing but an API token, and
telemetry is attached to it purely by the ``X-DataRobot-Entity-Id`` header on the
OTLP export, so no deployment, model, or workload has to exist first.

Three steps, each a plain function so the conftest stays a thin pytest wrapper:

1. :func:`provision_use_case` -- create the Use Case.
2. :func:`emit_failing_agent_run` -- OTLP/HTTP-export a small but realistically
   shaped agentic run into it: one failing trace (``agent.run`` -> tool call ->
   ``llm.chat`` that errors with a 429 and carries ``gen_ai.*`` prompt/completion
   attributes) plus one healthy trace, and an error-level log line correlated to
   the failing span. Every export result is recorded so an auth/entity/flag
   problem fails loudly instead of surfacing minutes later as a polling timeout.
3. :func:`wait_for_otel_ingestion` -- poll the same ``GET /otel/...`` REST endpoints
   the tools read until the failing trace, its ERROR span, and the error log are
   all visible. Ingestion is asynchronous (collector -> Elasticsearch), and the
   tests are about tool selection, not ingest latency, so the wait happens here.

:func:`cleanup_use_case` deletes the telemetry and the Use Case at session end.
"""

import dataclasses
import json
import logging
import re
import time
from collections.abc import Sequence
from typing import Any

from datarobot.errors import ClientError
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs import LoggingHandler
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import SpanKind
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode

# The OTel entity type of a Use Case, on the wire and in the tools' ``entity_type``.
OTEL_ACCEPTANCE_ENTITY_TYPE = "experiment_container"

# Prefix every provisioned Use Case carries, so a leftover from an interrupted run is
# recognisable in the UI and safe to delete by hand.
USE_CASE_NAME_PREFIX = "MCP OTel acceptance"

# Realistic agentic content. Kept short: these tests measure tool selection, not
# truncation -- the oversized-trace population is covered by the tier-1 fixtures.
_QUESTION = (
    "Which of my deployments are erroring right now, and does the Q3 renewal quote "
    "already include the negotiated multi-year discount?"
)
_TOOL_RESULT = (
    "14 rows returned from deployments_list, of which 3 are in an errored state and 1 "
    "has been stuck in provisioning for 41 minutes."
)
_PARTIAL_COMPLETION = (
    "Three deployments are currently erroring: churn-scorer-prod, renewal-quote-v2 and "
    "support-triage. The renewal-quote-v2 failure is the one that affects the Q3 quote, "
    "because that deployment computes the multi-year discount. Before I can confirm the "
    "corrected total I need to"
)
_OK_COMPLETION = (
    "All 14 deployments are healthy. The Q3 renewal quote already includes the "
    "negotiated 12% multi-year discount, so the total of $184,300 stands."
)
_LLM_ERROR = (
    "HTTPError 429 Too Many Requests from the LLM gateway after 3 retries with exponential backoff"
)


@dataclasses.dataclass(frozen=True)
class EmittedTelemetry:
    """What :func:`emit_failing_agent_run` sent, as the read API will identify it."""

    failing_trace_id: str
    failing_span_id: str
    ok_trace_id: str
    error_log_message: str
    run_label: str


@dataclasses.dataclass(frozen=True)
class OtelAcceptanceEntity:
    """The entity the OTel acceptance cases run against, provisioned or external."""

    entity_type: str
    entity_id: str
    failing_trace_id: str
    failing_span_id: str


def otel_ingest_base_url(api_endpoint: str) -> str:
    """Derive the OTLP ingest base (``https://host/otel``) from a ``/api/v2`` endpoint.

    The collector ingress lives at the same host as the REST API but off ``/otel``, not
    under ``/api/v2`` -- the same derivation the monitoring skill's
    ``create_use_case.py`` and this repo's ``core/telemetry/datarobot_otel.py`` make.
    """
    base = api_endpoint.rstrip("/").removesuffix("/api/v2")
    return f"{base}/otel"


def otel_ingest_headers(entity_type: str, entity_id: str, api_token: str) -> dict[str, str]:
    """Headers that route an OTLP export to one entity and authenticate it."""
    return {
        "X-DataRobot-Entity-Id": f"{entity_type}-{entity_id}",
        "X-DataRobot-Api-Key": api_token,
    }


def otel_api_field(row: dict[str, Any], name: str) -> Any:
    """Read one snake_case field from a *raw* ``/otel`` row, whichever case the API used.

    These helpers call the REST API directly rather than going through
    ``OtelQueryApiClient``, so they see drflask's camelCase on the wire (``traceId``,
    ``statusCode``) rather than the snake_case that client normalizes to. Accepting both
    spellings keeps them working on either side of that normalization -- and, more to the
    point, stops a spelling mismatch from being swallowed as a ``pytest.skip``.
    """
    if name in row:
        return row[name]
    return row.get(re.sub(r"_(\w)", lambda m: m.group(1).upper(), name))


# --- provisioning ---------------------------------------------------------------------


def provision_use_case(dr_module: Any, *, name: str, description: str) -> str:
    """Create a Use Case through the configured ``datarobot`` SDK; return its id."""
    use_case = dr_module.UseCase.create(name=name, description=description)
    return str(use_case.id)


def cleanup_use_case(dr_module: Any, rest_client: Any, use_case_id: str) -> list[str]:
    """Best-effort teardown: delete the entity's traces and logs, then the Use Case.

    Deleting the Use Case does not delete the telemetry attached to it (that lives in
    the OTel store, keyed by ``experiment_container-<id>``), so both are removed. Each
    step is independent -- a 403 on the OTel deletes must not leave the Use Case behind.
    Returns the failures as messages rather than raising: teardown runs in a session
    finalizer, where an exception would only mask the test results.
    """
    warnings: list[str] = []
    for signal in ("traces", "logs"):
        try:
            rest_client.delete(f"otel/{OTEL_ACCEPTANCE_ENTITY_TYPE}/{use_case_id}/{signal}/")
        except Exception as exc:  # noqa: BLE001 - reported, not raised, in a finalizer
            warnings.append(f"could not delete OTel {signal} for use case {use_case_id}: {exc}")
    try:
        dr_module.UseCase.delete(use_case_id)
    except Exception as exc:  # noqa: BLE001 - reported, not raised, in a finalizer
        warnings.append(f"could not delete use case {use_case_id}: {exc}")
    return warnings


# --- emission -------------------------------------------------------------------------


class _RecordingSpanExporter(OTLPSpanExporter):
    """OTLP span exporter that keeps every export result.

    ``SimpleSpanProcessor`` swallows export failures (it logs and moves on), so without
    this a bad token or a mistyped entity id would only show up as a polling timeout
    minutes later. Recording the results lets :func:`emit_failing_agent_run` fail at
    the export, with the HTTP status the exporter logged, instead.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.results: list[Any] = []
        self.errors: list[str] = []

    def export(self, spans: Sequence[Any]) -> Any:
        try:
            result = super().export(spans)
        except Exception as exc:  # noqa: BLE001 - recorded so the caller can raise
            self.errors.append(repr(exc))
            raise
        self.results.append(result)
        return result


class _RecordingLogExporter(OTLPLogExporter):
    """Log-record twin of :class:`_RecordingSpanExporter`."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.results: list[Any] = []
        self.errors: list[str] = []

    def export(self, batch: Sequence[Any]) -> Any:
        try:
            result = super().export(batch)
        except Exception as exc:  # noqa: BLE001 - recorded so the caller can raise
            self.errors.append(repr(exc))
            raise
        self.results.append(result)
        return result


def _hex_trace_id(span: Any) -> str:
    return format(span.get_span_context().trace_id, "032x")


def _hex_span_id(span: Any) -> str:
    return format(span.get_span_context().span_id, "016x")


def _set_gen_ai_prompt(span: Any, prompt: str) -> None:
    # Both the flat attribute the tracing UI's Prompt column reads and the indexed
    # ``gen_ai.prompt.N.*`` form the platform's enrich_trace folds into
    # ``gen_ai.input.messages`` -> the trace's ``prompt`` field.
    span.set_attribute("gen_ai.prompt", prompt)
    span.set_attribute("gen_ai.prompt.0.role", "user")
    span.set_attribute("gen_ai.prompt.0.content", prompt)


def _set_gen_ai_completion(span: Any, completion: str) -> None:
    span.set_attribute("gen_ai.completion", completion)
    span.set_attribute("gen_ai.completion.0.role", "assistant")
    span.set_attribute("gen_ai.completion.0.content", completion)
    # Traceloop-instrumented agents write the same text again under their own key; the
    # byte-identical twin is what the tools' dedup reports as ``dropped_as_duplicate``.
    span.set_attribute("traceloop.entity.output", completion)


def _emit_ok_run(tracer: Any, logger: logging.Logger, run_label: str) -> str:
    with tracer.start_as_current_span("agent.run", kind=SpanKind.SERVER) as root:
        root.set_attribute("datarobot.acceptance_test", run_label)
        _set_gen_ai_prompt(root, _QUESTION)
        logger.info("agent.run started [%s]", run_label)
        with tracer.start_as_current_span("tool.deployments_list", kind=SpanKind.INTERNAL) as tool:
            tool.set_attribute("tool_name", "deployments_list")
            tool.set_attribute("gen_ai.tool.name", "deployments_list")
            tool.set_attribute("tool.parameters", json.dumps({"limit": 20}))
            tool.set_attribute(
                "tool.result", "14 rows returned from deployments_list, all healthy."
            )
        with tracer.start_as_current_span("llm.chat", kind=SpanKind.CLIENT) as llm:
            llm.set_attribute("gen_ai.system", "datarobot-llm-gateway")
            llm.set_attribute("gen_ai.request.model", "gpt-4o")
            _set_gen_ai_prompt(llm, f"{_QUESTION}\n\nTool result: {_TOOL_RESULT}")
            _set_gen_ai_completion(llm, _OK_COMPLETION)
            llm.set_attribute("gen_ai.usage.prompt_tokens", 398)
            llm.set_attribute("gen_ai.usage.completion_tokens", 41)
        _set_gen_ai_completion(root, _OK_COMPLETION)
        logger.info("agent.run completed [%s]", run_label)
        return _hex_trace_id(root)


def _emit_failing_run(tracer: Any, logger: logging.Logger, run_label: str) -> tuple[str, str, str]:
    error_log_message = f"LLM gateway call failed; giving up after 3 retries [{run_label}]"
    with tracer.start_as_current_span("agent.run", kind=SpanKind.SERVER) as root:
        root.set_attribute("datarobot.acceptance_test", run_label)
        _set_gen_ai_prompt(root, _QUESTION)
        logger.info("agent.run started [%s]", run_label)
        with tracer.start_as_current_span("tool.deployments_list", kind=SpanKind.INTERNAL) as tool:
            tool.set_attribute("tool_name", "deployments_list")
            tool.set_attribute("gen_ai.tool.name", "deployments_list")
            tool.set_attribute("tool.parameters", json.dumps({"limit": 20}))
            tool.set_attribute("tool.result", _TOOL_RESULT)
        with tracer.start_as_current_span("llm.chat", kind=SpanKind.CLIENT) as llm:
            llm.set_attribute("gen_ai.system", "datarobot-llm-gateway")
            llm.set_attribute("gen_ai.request.model", "gpt-4o")
            _set_gen_ai_prompt(llm, f"{_QUESTION}\n\nTool result: {_TOOL_RESULT}")
            # The gateway streamed part of an answer before the 429 -- so the failing
            # span really does carry LLM output, which is what the drill-down case asks
            # for.
            _set_gen_ai_completion(llm, _PARTIAL_COMPLETION)
            llm.set_attribute("gen_ai.usage.prompt_tokens", 412)
            llm.set_attribute("gen_ai.usage.completion_tokens", 57)
            llm.set_attribute("llm.gateway.retries", 3)
            error = RuntimeError(_LLM_ERROR)
            llm.record_exception(error)
            llm.set_status(Status(StatusCode.ERROR, _LLM_ERROR))
            try:
                raise error
            except RuntimeError:
                # ``exception`` -> level ERROR with a stack trace, emitted inside the span
                # so the log record carries this trace_id/span_id for correlation.
                logger.exception(error_log_message)
            failing_span_id = _hex_span_id(llm)
        root.set_status(Status(StatusCode.ERROR, f"agent run failed: llm.chat raised {_LLM_ERROR}"))
        return _hex_trace_id(root), failing_span_id, error_log_message


def _check_exports(kind: str, results: Sequence[Any], errors: Sequence[str]) -> None:
    # Compared by member name, not identity: the span exporter returns
    # ``SpanExportResult`` while the log exporter returns ``LogRecordExportResult`` (of
    # which ``LogExportResult`` is a distinct, older enum), and both spell success the
    # same way.
    failed = [r for r in results if r.name != "SUCCESS"]
    if not results:
        raise RuntimeError(f"no {kind} were exported at all")
    if failed or errors:
        raise RuntimeError(
            f"{len(failed)} of {len(results)} {kind} exports failed"
            + (f" ({'; '.join(errors)})" if errors else "")
            + ". Common causes: an invalid DATAROBOT_API_TOKEN, an X-DataRobot-Entity-Id the "
            "token cannot access, or a cluster without the OTel ingest enabled. The exporter "
            "logged the HTTP status above."
        )


def emit_failing_agent_run(
    *, ingest_base_url: str, headers: dict[str, str], run_label: str
) -> EmittedTelemetry:
    """OTLP-export one failing and one healthy agentic run, plus correlated logs.

    Uses private ``TracerProvider``/``LoggerProvider`` instances (never the OTel globals,
    which other code in the pytest process may own) with simple, synchronous processors,
    so by the time this returns every span and log record has been exported and its
    result recorded. Raises ``RuntimeError`` if any export failed.
    """
    resource = Resource.create({"datarobot.acceptance_test": run_label})

    span_exporter = _RecordingSpanExporter(
        endpoint=f"{ingest_base_url}/v1/traces", headers=headers, timeout=30
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    tracer = tracer_provider.get_tracer("datarobot_genai.tests.acceptance.otel")

    log_exporter = _RecordingLogExporter(
        endpoint=f"{ingest_base_url}/v1/logs", headers=headers, timeout=30
    )
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    # ``opentelemetry-sdk`` deprecates this handler in favour of
    # ``opentelemetry-instrumentation-logging``, which this repo does not depend on. It
    # is still what the monitoring skill's generated ``configure_otel()`` and the
    # platform's own OTel acceptance tests use, so the records it produces (body,
    # severity, exception stack trace, current-span correlation) are the shape real
    # instrumented agents send.
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
    logger = logging.getLogger(f"datarobot_genai.tests.acceptance.otel.{run_label}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        ok_trace_id = _emit_ok_run(tracer, logger, run_label)
        failing_trace_id, failing_span_id, error_log_message = _emit_failing_run(
            tracer, logger, run_label
        )
    finally:
        logger.removeHandler(handler)
        tracer_provider.shutdown()
        logger_provider.shutdown()

    _check_exports("span", span_exporter.results, span_exporter.errors)
    _check_exports("log", log_exporter.results, log_exporter.errors)
    return EmittedTelemetry(
        failing_trace_id=failing_trace_id,
        failing_span_id=failing_span_id,
        ok_trace_id=ok_trace_id,
        error_log_message=error_log_message,
        run_label=run_label,
    )


# --- read-back ------------------------------------------------------------------------


def list_error_trace_ids(rest_client: Any, entity_type: str, entity_id: str) -> list[str]:
    """``trace_id``s of the newest error-status traces on the entity (raw REST call)."""
    result = (
        rest_client.get(
            f"otel/{entity_type}/{entity_id}/traces/", params={"status": "error", "limit": 50}
        )
    ).json()
    rows = result.get("traces") or result.get("data") or []
    return [str(otel_api_field(row, "trace_id")) for row in rows]


def list_error_span_ids(
    rest_client: Any, entity_type: str, entity_id: str, trace_id: str
) -> list[str]:
    """``span_id``s of the ERROR-status spans in one trace (first 100 spans, raw REST).

    The comparison is case-insensitive: the API reports ``"Error"``, not the ``"ERROR"``
    OTel itself uses. A 404 (trace not ingested yet) reads as "no error spans yet".
    """
    try:
        result = rest_client.get(
            f"otel/{entity_type}/{entity_id}/traces/{trace_id}/", params={"limit": 100}
        ).json()
    except ClientError as exc:
        if exc.status_code == 404:  # not ingested yet
            return []
        raise
    return [
        str(otel_api_field(span, "span_id"))
        for span in result.get("spans") or []
        if str(otel_api_field(span, "status_code") or "").upper() == "ERROR"
    ]


def list_error_log_messages(
    rest_client: Any, entity_type: str, entity_id: str, *, includes: str
) -> list[str]:
    """Messages of error-level log lines containing ``includes`` (raw REST call)."""
    result = rest_client.get(
        f"otel/{entity_type}/{entity_id}/logs/",
        params={"level": "error", "limit": 50, "includes": includes},
    ).json()
    return [str(row.get("message") or "") for row in result.get("data") or []]


def wait_for_otel_ingestion(
    rest_client: Any,
    entity_type: str,
    entity_id: str,
    emitted: EmittedTelemetry,
    *,
    timeout_s: float = 180.0,
    interval_s: float = 3.0,
) -> float:
    """Block until the emitted failing trace, its ERROR span, and the error log are readable.

    Polls the three REST reads the acceptance cases' tools depend on, dropping each
    condition as it becomes true, so a partial ingest (traces landed, logs did not) is
    named precisely in the ``TimeoutError``. Returns the seconds it took. Any
    ``ClientError`` from the reads propagates -- a 403 here is a permissions/feature-flag
    problem the caller should report as such, not retry.
    """
    started = time.monotonic()
    deadline = started + timeout_s
    pending = {
        "error trace listed by GET traces/?status=error",
        "ERROR span visible in GET traces/{trace_id}/",
        "error log visible in GET logs/?level=error",
    }
    while True:
        if "error trace listed by GET traces/?status=error" in pending and (
            emitted.failing_trace_id in list_error_trace_ids(rest_client, entity_type, entity_id)
        ):
            pending.discard("error trace listed by GET traces/?status=error")
        if "ERROR span visible in GET traces/{trace_id}/" in pending and (
            emitted.failing_span_id
            in list_error_span_ids(rest_client, entity_type, entity_id, emitted.failing_trace_id)
        ):
            pending.discard("ERROR span visible in GET traces/{trace_id}/")
        if "error log visible in GET logs/?level=error" in pending and any(
            emitted.run_label in message
            for message in list_error_log_messages(
                rest_client, entity_type, entity_id, includes=emitted.run_label
            )
        ):
            pending.discard("error log visible in GET logs/?level=error")
        if not pending:
            return time.monotonic() - started
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"OTel data for {entity_type}/{entity_id} (trace {emitted.failing_trace_id}) "
                f"was not readable {timeout_s:.0f}s after export; still missing: "
                f"{sorted(pending)}"
            )
        time.sleep(interval_s)
