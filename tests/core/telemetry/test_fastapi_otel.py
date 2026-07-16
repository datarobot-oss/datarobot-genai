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
from dataclasses import dataclass

import pytest

from datarobot_genai.core.telemetry.fastapi_otel import DEFAULT_EXCLUDED_TRACE_SPAN_NAMES
from datarobot_genai.core.telemetry.fastapi_otel import OTel

_ENV_KEYS_TO_CLEAR = ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_HEADERS")

_INSTRUMENTORS = []
for _module_path, _class_name in (
    ("opentelemetry.instrumentation.requests", "RequestsInstrumentor"),
    ("opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
    ("opentelemetry.instrumentation.sqlalchemy", "SQLAlchemyInstrumentor"),
):
    try:
        _module = __import__(_module_path, fromlist=[_class_name])
        _INSTRUMENTORS.append(getattr(_module, _class_name))
    except ImportError:
        pass


@dataclass
class FakeOTelConfig:
    otel_exporter_otlp_endpoint: str = ""
    otel_exporter_otlp_headers: str = ""
    otel_sdk_disabled: bool = False


def _reset_otel_singleton() -> None:
    OTel._instance = None
    OTel._initialized = False
    OTel._auto_instrumentation_setup = False


def _uninstrument_all() -> None:
    for instrumentor_cls in _INSTRUMENTORS:
        instrumentor = instrumentor_cls()
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@pytest.fixture(autouse=True)
def reset_otel_singleton(monkeypatch):
    """Undo the process-wide side effects OTel.configure()/_setup_auto_instrumentation
    leave behind: the singleton, env vars mirrored from config, and global
    library auto-instrumentation. Note: the global OTel TracerProvider/
    MeterProvider/LoggerProvider set via trace.set_tracer_provider() etc. have
    no public "unset" API and are NOT reset here - a test that configures one
    leaves it installed process-wide for the rest of the test session. What
    IS handled: shutdown() is called on the current instance before it's
    discarded, so any PeriodicExportingMetricReader/BatchSpanProcessor
    background threads it started are stopped rather than left retrying
    against localhost:4318 for the rest of the process's life.
    """
    _reset_otel_singleton()
    for key in _ENV_KEYS_TO_CLEAR:
        monkeypatch.delenv(key, raising=False)
    yield
    if OTel._instance is not None:
        OTel._instance.shutdown()
    _reset_otel_singleton()
    for key in _ENV_KEYS_TO_CLEAR:
        monkeypatch.delenv(key, raising=False)
    _uninstrument_all()


def test_otel_is_a_singleton() -> None:
    assert OTel() is OTel()


def test_configure_disabled_via_sdk_disabled() -> None:
    otel = OTel()
    otel.configure(FakeOTelConfig(otel_sdk_disabled=True))
    assert otel.telemetry_enabled is False


def test_configure_disables_telemetry_without_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT_INTERNAL", raising=False)
    otel = OTel()
    otel.configure(FakeOTelConfig(otel_exporter_otlp_endpoint=""))
    assert otel.telemetry_enabled is False


def test_configure_disables_telemetry_for_remote_endpoint_without_headers() -> None:
    otel = OTel()
    otel.configure(
        FakeOTelConfig(
            otel_exporter_otlp_endpoint="https://otel.datarobot.com",
            otel_exporter_otlp_headers="",
        )
    )
    assert otel.telemetry_enabled is False


def test_configure_enables_telemetry_for_local_endpoint_without_headers() -> None:
    otel = OTel()
    otel.configure(
        FakeOTelConfig(
            otel_exporter_otlp_endpoint="http://localhost:4318",
            otel_exporter_otlp_headers="",
        )
    )
    assert otel.telemetry_enabled is True


def test_configure_enables_telemetry_for_remote_endpoint_with_headers() -> None:
    otel = OTel()
    otel.configure(
        FakeOTelConfig(
            otel_exporter_otlp_endpoint="https://otel.datarobot.com",
            otel_exporter_otlp_headers="x-datarobot-api-key=abc",
        )
    )
    assert otel.telemetry_enabled is True


def test_excluded_trace_span_names_empty_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OTEL_EXCLUDED_TRACE_SPAN_NAMES", raising=False)
    otel = OTel()
    assert DEFAULT_EXCLUDED_TRACE_SPAN_NAMES == frozenset()
    assert otel._get_excluded_trace_span_names() == frozenset()


def test_excluded_trace_span_names_reads_env_override(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_EXCLUDED_TRACE_SPAN_NAMES", "app.noisy_span, app.other")
    otel = OTel()
    names = otel._get_excluded_trace_span_names()
    assert names == {"app.noisy_span", "app.other"}


def test_trace_decorator_skips_excluded_span_name(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_EXCLUDED_TRACE_SPAN_NAMES", "app.noisy_span")
    otel = OTel()

    def handler() -> str:
        return "ok"

    wrapped = otel.trace("app.noisy_span")(handler)

    # An excluded span name means the function is returned unwrapped.
    assert wrapped is handler
    assert wrapped() == "ok"


def test_shutdown_without_configuring_is_a_no_op() -> None:
    otel = OTel()
    otel.shutdown()  # must not raise


def test_shutdown_calls_provider_shutdown_once_configured() -> None:
    otel = OTel()
    otel.configure(FakeOTelConfig(otel_exporter_otlp_endpoint="http://localhost:4318"))
    otel.configure_tracing()
    assert otel._tracer_provider is not None
    otel.shutdown()


def test_trace_decorator_wraps_and_creates_span() -> None:
    otel = OTel()
    otel.configure(FakeOTelConfig(otel_exporter_otlp_endpoint="http://localhost:4318"))

    calls = []

    @otel.trace
    def handler(value: int) -> int:
        calls.append(value)
        return value * 2

    assert handler is not None
    assert handler(21) == 42
    assert calls == [21]


async def test_trace_decorator_wraps_async_function() -> None:
    otel = OTel()
    otel.configure(FakeOTelConfig(otel_exporter_otlp_endpoint="http://localhost:4318"))

    @otel.trace
    async def handler() -> str:
        return "ok"

    assert await handler() == "ok"


# Module-level with a short name: the `meter` decorator builds a metric
# name as f"function.{func.__module__}.{func.__qualname__}", capped at
# OTel's 63-char instrument name limit. A function nested inside a test
# picks up "<locals>" in its qualname and blows that budget immediately -
# a real, pre-existing constraint of this decorator, not something to
# route around inside the test body.
def _m() -> str:
    return "ok"


def _m_raises() -> None:
    raise ValueError("boom")


def test_meter_decorator_records_without_raising() -> None:
    otel = OTel()
    otel.configure(FakeOTelConfig(otel_exporter_otlp_endpoint="http://localhost:4318"))
    assert otel.meter(_m)() == "ok"


def test_meter_decorator_records_on_exception() -> None:
    otel = OTel()
    otel.configure(FakeOTelConfig(otel_exporter_otlp_endpoint="http://localhost:4318"))
    with pytest.raises(ValueError, match="boom"):
        otel.meter(_m_raises)()


def test_span_context_manager_sets_attributes() -> None:
    otel = OTel()
    otel.configure(FakeOTelConfig(otel_exporter_otlp_endpoint="http://localhost:4318"))

    with otel.span("my-span", key="value") as active_span:
        assert active_span is not None
