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

"""Wire-level e2e for the sandbox SLI metrics pipeline.

The dragent server exports *traces* to the session mock collector; nothing on
the agent side emits metrics. This test drives the real sandbox
instrumentation (``InstrumentedSandbox`` + the SLI instrument definitions from
``build_instruments``) through a real ``OTLPMetricExporter`` over HTTP into
the same collector, proving the metrics leg end to end: SLI emission →
OTLP/HTTP wire (with the DataRobot auth headers) → collector ingest → parse.

``drmcpbase.bootstrap_metrics_provider`` (the production provider bootstrap)
is not importable in this env — ``drmcpbase.__init__`` needs fastmcp, which
the dragent extras don't install — so the provider is assembled from the same
SDK pieces the bootstrap uses. The bootstrap itself is unit-tested in
``tests/drmcpbase`` and was verified against a live staging deployment.
"""

import asyncio
from typing import Any

import pytest
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

from datarobot_genai.drtools.core.sandbox.base import SandboxError
from datarobot_genai.drtools.core.sandbox.base import SandboxResult
from datarobot_genai.drtools.core.sandbox.observability import InstrumentedSandbox
from datarobot_genai.drtools.core.sandbox.observability import build_instruments

from .otel_helpers import OTEL_API_KEY
from .otel_helpers import OTEL_ENTITY_ID
from .otel_helpers import OTEL_EXPORTER_OTLP_ENDPOINT
from .otel_helpers import OTLP_METRICS_PATH
from .otel_helpers import MockOtelCollector


class _StubSandbox:
    """Minimal Sandbox Protocol impl — the SLIs are backend-agnostic."""

    def __init__(self, *, result: SandboxResult | None = None, error: Exception | None = None):
        self._result = result
        self._error = error

    async def run(
        self,
        code: str,
        *,
        inputs: dict[str, Any] | None = None,
        externals: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> SandboxResult:
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def test_sandbox_sli_metrics_reach_collector(otel_collector: MockOtelCollector) -> None:
    exporter = OTLPMetricExporter(
        endpoint=f"{OTEL_EXPORTER_OTLP_ENDPOINT}/v1/metrics",
        headers={
            "X-DataRobot-Api-Key": OTEL_API_KEY,
            "X-DataRobot-Entity-Id": OTEL_ENTITY_ID,
        },
    )
    # Long interval: the test controls flush timing via force_flush() below.
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=60_000)
    # Deliberately NOT installed as the global provider — dependency-injected
    # instruments keep this test isolated from any other OTel state in the run.
    provider = MeterProvider(metric_readers=[reader])
    instruments = build_instruments(provider.get_meter("e2e-sandbox-sli"))

    ok = SandboxResult(stdout="ok", stderr="", return_value=1, duration_s=0.1, exit_code=0)
    asyncio.run(InstrumentedSandbox(_StubSandbox(result=ok), instruments=instruments).run("code"))
    with pytest.raises(SandboxError):
        # 137 = 128 + SIGKILL(9): how an OOM-killed container surfaces.
        asyncio.run(
            InstrumentedSandbox(
                _StubSandbox(error=SandboxError("killed", exit_code=137)),
                instruments=instruments,
            ).run("code")
        )

    try:
        provider.force_flush()

        totals = otel_collector.wait_for_metrics(lambda m: m.name == "sandbox.execution_total")
        outcomes = {m.attributes.get("outcome") for m in totals}
        assert {"success", "failure"} <= outcomes, f"got outcomes {outcomes} in {totals}"

        failures = otel_collector.wait_for_metrics(
            lambda m: m.name == "sandbox.execution_failure_total"
        )
        assert any(m.attributes.get("reason") == "oom" for m in failures), failures

        durations = otel_collector.wait_for_metrics(
            lambda m: m.name == "sandbox.execution_duration_seconds"
        )
        assert any(m.kind == "histogram" and m.value >= 1 for m in durations), durations

        # The DataRobot ingest auth headers must reach the collector unmodified.
        metric_posts = [r for r in otel_collector.requests if r.path == OTLP_METRICS_PATH]
        assert metric_posts, "no OTLP metrics POST captured"
        headers = {k.lower(): v for k, v in metric_posts[-1].headers.items()}
        assert headers.get("x-datarobot-api-key") == OTEL_API_KEY
        assert headers.get("x-datarobot-entity-id") == OTEL_ENTITY_ID
    finally:
        provider.shutdown()
