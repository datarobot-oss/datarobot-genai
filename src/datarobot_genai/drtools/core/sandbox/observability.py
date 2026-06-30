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

"""Observability for sandboxed code execution.

Defines the SLI building blocks: a pure :func:`classify_outcome` that turns an
execution result/error into an ``(outcome, reason)`` pair, plus (later in this
module) the OTel instruments and the :class:`InstrumentedSandbox` wrapper.

:func:`classify_outcome` is intentionally free of any OpenTelemetry import so it
remains usable — and unit-testable — in environments where the optional OTel
stack (the ``dragent`` extra) is not installed.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from datarobot_genai.drtools.core.sandbox.base import SandboxInfraError
from datarobot_genai.drtools.core.sandbox.base import SandboxResult
from datarobot_genai.drtools.core.sandbox.base import SandboxTimeout

logger = logging.getLogger(__name__)

# Meter name; dotted, namespaced under the package (OTel house style). Metric
# names mirror the workload-api convention so the SLIs read the same on either
# side of the genai-client / platform boundary.
METER_NAME = "datarobot_genai.sandbox"

# 128 + SIGKILL(9): how a container the kernel OOM-kills surfaces its exit code.
_OOM_EXIT_CODE = 137

# Substrings in stderr that indicate an out-of-memory condition even when the
# exit code is ambiguous (e.g. CPython raising MemoryError before being killed).
_OOM_MARKERS = ("OOMKilled", "MemoryError")

# Outcome / reason label values. ``outcome`` is the success-rate dimension;
# ``reason`` is the failure-taxonomy dimension (only set on failure).
OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"


def classify_outcome(error: BaseException | None) -> tuple[str, str | None]:
    """Classify an execution into ``(outcome, reason)`` for the SLIs.

    Parameters
    ----------
    error
        The exception raised by the sandbox, or ``None`` if the execution
        returned a :class:`~datarobot_genai.drtools.core.sandbox.base.SandboxResult`.

    Returns
    -------
    tuple[str, str | None]
        ``("success", None)`` on success, otherwise ``("failure", reason)``
        where ``reason`` is one of ``"timeout"``, ``"oom"``, ``"infra"``, or
        ``"crash"``.
    """
    if error is None:
        return OUTCOME_SUCCESS, None
    if isinstance(error, SandboxTimeout):
        return OUTCOME_FAILURE, "timeout"
    if isinstance(error, SandboxInfraError):
        return OUTCOME_FAILURE, "infra"

    exit_code = getattr(error, "exit_code", None)
    stderr = getattr(error, "stderr", "") or ""
    if exit_code == _OOM_EXIT_CODE or any(marker in stderr for marker in _OOM_MARKERS):
        return OUTCOME_FAILURE, "oom"
    return OUTCOME_FAILURE, "crash"


@dataclass
class SandboxInstruments:
    """The three OTel instruments backing the sandbox SLIs.

    Held together so they can be dependency-injected (tests bind them to an
    in-memory reader) or lazily built from the global meter in production.
    """

    execution_total: Any  # Counter: denominator + success-rate numerator
    execution_duration: Any  # Histogram (seconds): end-to-end latency SLI
    execution_failures: Any  # Counter: failure-taxonomy SLI (by reason)


def build_instruments(meter: Any) -> SandboxInstruments:
    """Create the sandbox instruments on ``meter`` (an OTel ``Meter``)."""
    return SandboxInstruments(
        execution_total=meter.create_counter(
            "sandbox.execution_total",
            unit="1",
            description="Sandbox executions, labeled by outcome (success|failure).",
        ),
        execution_duration=meter.create_histogram(
            "sandbox.execution_duration_seconds",
            unit="s",
            description="End-to-end sandbox execution latency, labeled by outcome.",
            # Fine sub-second→30s boundaries: sandbox executions are short, so the
            # SDK default buckets (0,5,10,25,...) collapse everything into one bucket
            # and make p95 meaningless. These give an informative latency SLI.
            explicit_bucket_boundaries_advisory=[
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.0,
                3.0,
                5.0,
                10.0,
                20.0,
                30.0,
            ],
        ),
        execution_failures=meter.create_counter(
            "sandbox.execution_failure_total",
            unit="1",
            description="Sandbox failures, labeled by reason (timeout|oom|infra|crash).",
        ),
    )


# Lazily-initialized singletons. A mutable dict avoids the ``global`` statement
# (mirrors the ``_BOOTSTRAP_STATE`` pattern in ``core/datarobot_otel.py``).
_STATE: dict[str, Any] = {"instruments": None, "provider": None, "tracer": None}


def get_instruments() -> SandboxInstruments | None:
    """Lazily build instruments from the global meter; ``None`` if OTel absent.

    Instruments are cached, but keyed on the active global ``MeterProvider``: if
    the provider changes (e.g. a no-op/proxy provider was current on first use
    and ``bootstrap_metrics_provider`` later installs the real SDK provider), the
    instruments are rebuilt against the new provider rather than staying pinned
    to the old one. Returns ``None`` (callers no-op) when the optional
    OpenTelemetry stack is not installed, keeping ``drtools`` usable without the
    ``dragent`` extra.
    """
    try:
        from opentelemetry import metrics
    except ImportError:
        return None
    provider = metrics.get_meter_provider()
    if _STATE["instruments"] is None or _STATE["provider"] is not provider:
        _STATE["instruments"] = build_instruments(metrics.get_meter(METER_NAME))
        _STATE["provider"] = provider
    return _STATE["instruments"]


def record_execution(
    outcome: str,
    reason: str | None,
    duration_s: float,
    *,
    instruments: SandboxInstruments | None = None,
) -> None:
    """Emit the SLI metrics for one execution. No-op if OTel is unavailable."""
    instruments = instruments or get_instruments()
    if instruments is None:
        return
    outcome_attrs = {"outcome": outcome}
    instruments.execution_total.add(1, outcome_attrs)
    instruments.execution_duration.record(duration_s, outcome_attrs)
    if outcome == OUTCOME_FAILURE and reason:
        instruments.execution_failures.add(1, {"reason": reason})


def get_tracer() -> Any | None:
    """Lazily fetch the global tracer; ``None`` if OpenTelemetry is absent."""
    if _STATE["tracer"] is None:
        try:
            from opentelemetry import trace
        except ImportError:
            return None
        _STATE["tracer"] = trace.get_tracer(METER_NAME)
    return _STATE["tracer"]


def _mark_span_error(span: Any, reason: str | None) -> None:
    """Set the failure attributes + ERROR status on ``span`` (best effort)."""
    span.set_attribute("sandbox.outcome", OUTCOME_FAILURE)
    if reason:
        span.set_attribute("sandbox.failure_reason", reason)
    try:
        from opentelemetry.trace import Status
        from opentelemetry.trace import StatusCode

        span.set_status(Status(StatusCode.ERROR))
    except ImportError:  # pragma: no cover - OTel always present once span exists
        pass


class InstrumentedSandbox:
    """Wraps any :class:`~datarobot_genai.drtools.core.sandbox.base.Sandbox` backend
    and records SLI metrics + a span around each execution.

    Because it composes the ``Sandbox`` Protocol rather than subclassing a
    concrete backend, the same instrumentation covers ``LocalDockerSandbox``,
    ``DataRobotWorkloadSandbox``, and any future backend — define the SLIs once.
    """

    def __init__(
        self,
        inner: Any,
        *,
        instruments: SandboxInstruments | None = None,
        tracer: Any | None = None,
    ) -> None:
        self._inner = inner
        self._instruments = instruments
        self._tracer = tracer

    async def run(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        externals: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> SandboxResult:
        tracer = self._tracer or get_tracer()
        span_cm = (
            tracer.start_as_current_span("sandbox.execute")
            if tracer is not None
            else contextlib.nullcontext()
        )
        start = time.monotonic()
        with span_cm as span:
            try:
                result = await self._inner.run(
                    code, inputs=inputs, externals=externals, timeout_s=timeout_s
                )
            except Exception as error:
                # Exception (not BaseException) so CancelledError propagates
                # without being miscounted as a sandbox failure.
                outcome, reason = classify_outcome(error)
                record_execution(
                    outcome, reason, time.monotonic() - start, instruments=self._instruments
                )
                if span is not None:
                    _mark_span_error(span, reason)
                raise
            record_execution(
                OUTCOME_SUCCESS, None, time.monotonic() - start, instruments=self._instruments
            )
            if span is not None:
                span.set_attribute("sandbox.outcome", OUTCOME_SUCCESS)
            return result
