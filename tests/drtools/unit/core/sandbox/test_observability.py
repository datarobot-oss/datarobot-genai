# Copyright 2026 DataRobot, Inc.
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

"""Tests for sandbox observability.

Covers failure classification, metric emission, and the InstrumentedSandbox
wrapper.
"""

from typing import Any
from unittest.mock import MagicMock

import opentelemetry.metrics as om
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from datarobot_genai.drtools.core.sandbox import observability as obs
from datarobot_genai.drtools.core.sandbox.base import SandboxError
from datarobot_genai.drtools.core.sandbox.base import SandboxInfraError
from datarobot_genai.drtools.core.sandbox.base import SandboxResult
from datarobot_genai.drtools.core.sandbox.base import SandboxTimeout
from datarobot_genai.drtools.core.sandbox.observability import InstrumentedSandbox
from datarobot_genai.drtools.core.sandbox.observability import classify_outcome


def test_success_when_no_error() -> None:
    assert classify_outcome(None) == ("success", None)


def test_timeout_maps_to_timeout_reason() -> None:
    assert classify_outcome(SandboxTimeout("exceeded")) == ("failure", "timeout")


def test_oom_from_exit_code_137() -> None:
    # 137 = 128 + SIGKILL(9); how an OOM-killed container surfaces.
    assert classify_outcome(SandboxError("killed", exit_code=137)) == ("failure", "oom")


def test_oom_from_stderr_marker_even_with_other_exit_code() -> None:
    err = SandboxError("boom", exit_code=1, stderr="Traceback...\nMemoryError\n")
    assert classify_outcome(err) == ("failure", "oom")


def test_generic_nonzero_exit_is_crash() -> None:
    assert classify_outcome(SandboxError("boom", exit_code=1)) == ("failure", "crash")


def test_provision_transport_error_is_infra() -> None:
    assert classify_outcome(SandboxInfraError("docker not found")) == ("failure", "infra")


def test_sandbox_error_defaults_are_safe() -> None:
    # Raised without structured detail (older call sites) → still classifiable.
    assert classify_outcome(SandboxError("boom")) == ("failure", "crash")


# --- metric emission + wrapper -------------------------------------------------


def _reader_and_instruments() -> tuple[Any, Any]:
    """Build a fresh InMemoryMetricReader + instruments on their own provider.

    Dependency-injected so tests never touch the global MeterProvider (which
    OpenTelemetry only allows setting once per process).
    """
    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader]).get_meter("test")
    return reader, obs.build_instruments(meter)


def _points(reader: Any, name: str) -> list[Any]:
    data = reader.get_metrics_data()
    out: list[Any] = []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    out.extend(metric.data.data_points)
    return out


def test_record_success_emits_total_and_duration_no_failure() -> None:
    reader, instruments = _reader_and_instruments()
    obs.record_execution("success", None, 1.5, instruments=instruments)

    total = _points(reader, "sandbox.execution_total")
    assert any(p.attributes.get("outcome") == "success" and p.value == 1 for p in total)

    dur = _points(reader, "sandbox.execution_duration_seconds")
    assert any(p.attributes.get("outcome") == "success" and p.count == 1 for p in dur)

    # No failure counter datapoint should be created on success.
    assert _points(reader, "sandbox.execution_failure_total") == []


def test_record_failure_emits_failure_with_reason() -> None:
    reader, instruments = _reader_and_instruments()
    obs.record_execution("failure", "oom", 0.3, instruments=instruments)

    total = _points(reader, "sandbox.execution_total")
    assert any(p.attributes.get("outcome") == "failure" for p in total)

    failures = _points(reader, "sandbox.execution_failure_total")
    assert any(p.attributes.get("reason") == "oom" and p.value == 1 for p in failures)


class _FakeSandbox:
    """Minimal Sandbox Protocol impl for wrapper tests."""

    def __init__(self, *, result: SandboxResult | None = None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        externals: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> SandboxResult:
        self.calls.append({"code": code, "inputs": inputs, "timeout_s": timeout_s})
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


async def test_wrapper_passes_through_result_and_records_success() -> None:
    reader, instruments = _reader_and_instruments()
    result = SandboxResult(stdout="ok", stderr="", return_value=42, duration_s=0.1, exit_code=0)
    inner = _FakeSandbox(result=result)
    sandbox = InstrumentedSandbox(inner, instruments=instruments)

    returned = await sandbox.run("code", inputs={"a": 1}, timeout_s=5.0)

    assert returned is result
    assert inner.calls == [{"code": "code", "inputs": {"a": 1}, "timeout_s": 5.0}]
    total = _points(reader, "sandbox.execution_total")
    assert any(p.attributes.get("outcome") == "success" for p in total)


async def test_wrapper_classifies_oom_and_reraises() -> None:
    reader, instruments = _reader_and_instruments()
    inner = _FakeSandbox(error=SandboxError("killed", exit_code=137))
    sandbox = InstrumentedSandbox(inner, instruments=instruments)

    with pytest.raises(SandboxError):
        await sandbox.run("code")

    failures = _points(reader, "sandbox.execution_failure_total")
    assert any(p.attributes.get("reason") == "oom" for p in failures)


async def test_wrapper_records_timeout_reason() -> None:
    reader, instruments = _reader_and_instruments()
    inner = _FakeSandbox(error=SandboxTimeout("too slow"))
    sandbox = InstrumentedSandbox(inner, instruments=instruments)

    with pytest.raises(SandboxTimeout):
        await sandbox.run("code")

    failures = _points(reader, "sandbox.execution_failure_total")
    assert any(p.attributes.get("reason") == "timeout" for p in failures)


def _tracer_and_exporter() -> tuple[Any, Any]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


async def test_wrapper_emits_span_with_outcome_attribute_on_success() -> None:
    tracer, exporter = _tracer_and_exporter()
    _, instruments = _reader_and_instruments()
    result = SandboxResult(stdout="", stderr="", return_value=None, duration_s=0.0, exit_code=0)
    sandbox = InstrumentedSandbox(
        _FakeSandbox(result=result), instruments=instruments, tracer=tracer
    )

    await sandbox.run("code")

    spans = exporter.get_finished_spans()
    assert any(
        s.name == "sandbox.execute" and s.attributes.get("sandbox.outcome") == "success"
        for s in spans
    )


async def test_wrapper_span_records_failure_reason() -> None:
    tracer, exporter = _tracer_and_exporter()
    _, instruments = _reader_and_instruments()
    inner = _FakeSandbox(error=SandboxError("killed", exit_code=137))
    sandbox = InstrumentedSandbox(inner, instruments=instruments, tracer=tracer)

    with pytest.raises(SandboxError):
        await sandbox.run("code")

    spans = exporter.get_finished_spans()
    span = next(s for s in spans if s.name == "sandbox.execute")
    assert span.attributes.get("sandbox.outcome") == "failure"
    assert span.attributes.get("sandbox.failure_reason") == "oom"


def test_get_instruments_rebuilds_when_meter_provider_changes(monkeypatch) -> None:
    # Guards against pinning a no-op/proxy provider: if get_instruments() runs
    # before bootstrap installs the SDK provider, a later provider swap must
    # rebuild the instruments rather than keep emitting to the old provider.
    saved = dict(obs._STATE)
    try:
        obs._STATE["instruments"] = None
        obs._STATE["provider"] = None
        provider_a, provider_b = object(), object()
        seq = iter([provider_a, provider_a, provider_b])
        monkeypatch.setattr(om, "get_meter_provider", lambda: next(seq))
        monkeypatch.setattr(om, "get_meter", lambda _name: MagicMock())

        first = obs.get_instruments()
        cached = obs.get_instruments()  # same provider → cached
        rebuilt = obs.get_instruments()  # provider changed → rebuilt

        assert first is cached
        assert rebuilt is not first
    finally:
        obs._STATE.clear()
        obs._STATE.update(saved)
