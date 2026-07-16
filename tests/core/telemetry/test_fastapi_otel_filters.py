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
import logging

from datarobot_genai.core.telemetry.fastapi_otel import OTLPConnectionErrorFilter
from datarobot_genai.core.telemetry.fastapi_otel import _otel_handler_active
from datarobot_genai.core.telemetry.fastapi_otel import _SafeLoggingHandler


def _make_record(
    name: str, message: str, level: int = logging.ERROR, exc_info: object = None
) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=exc_info,
    )


def test_suppresses_urllib3_otlp_port_connection_error() -> None:
    record = _make_record(
        "urllib3.connectionpool",
        "HTTPConnectionPool(host='localhost', port=4318): Max retries exceeded "
        "with url: /v1/traces (Caused by NewConnectionError)",
    )
    assert OTLPConnectionErrorFilter().filter(record) is False


def test_suppresses_urllib3_error_matching_on_port_in_url() -> None:
    record = _make_record(
        "urllib3.connectionpool",
        "HTTPConnectionPool(host='localhost', port=4318): Max retries exceeded "
        "with url: http://localhost:4318/v1/metrics",
    )
    assert OTLPConnectionErrorFilter().filter(record) is False


def test_allows_urllib3_error_for_unrelated_port() -> None:
    record = _make_record(
        "urllib3.connectionpool",
        "HTTPConnectionPool(host='example.com', port=443): Max retries exceeded "
        "with url: /some/other/path",
    )
    assert OTLPConnectionErrorFilter().filter(record) is True


def test_suppresses_requests_connection_error_to_otlp_port() -> None:
    record = _make_record(
        "requests.adapters",
        "ConnectionError: connection refused to http://localhost:4318/v1/traces",
    )
    assert OTLPConnectionErrorFilter().filter(record) is False


def test_allows_requests_error_unrelated_to_otlp() -> None:
    record = _make_record("requests.adapters", "ConnectionError to example.com:443")
    assert OTLPConnectionErrorFilter().filter(record) is True


def test_suppresses_sdk_export_error_wrapping_connection_error() -> None:
    try:
        raise ConnectionError("refused")
    except ConnectionError:
        import sys

        exc_info = sys.exc_info()
    record = _make_record("opentelemetry.sdk.trace.export", "export failed", exc_info=exc_info)
    assert OTLPConnectionErrorFilter().filter(record) is False


def test_suppresses_sdk_export_error_via_chained_cause() -> None:
    try:
        try:
            raise ConnectionError("refused")
        except ConnectionError as inner:
            raise RuntimeError("export failed") from inner
    except RuntimeError:
        import sys

        exc_info = sys.exc_info()
    record = _make_record("opentelemetry.sdk.trace.export", "export failed", exc_info=exc_info)
    assert OTLPConnectionErrorFilter().filter(record) is False


def test_allows_sdk_error_unrelated_to_connection_failure() -> None:
    try:
        raise ValueError("something else")
    except ValueError:
        import sys

        exc_info = sys.exc_info()
    record = _make_record("opentelemetry.sdk.trace.export", "export failed", exc_info=exc_info)
    assert OTLPConnectionErrorFilter().filter(record) is True


def test_warning_callback_invoked_when_suppressing() -> None:
    calls = []
    otlp_filter = OTLPConnectionErrorFilter(warning_callback=lambda: calls.append(1))
    record = _make_record(
        "urllib3.connectionpool",
        "HTTPConnectionPool(host='localhost', port=4318): refused, url: /v1/traces",
    )
    otlp_filter.filter(record)
    assert calls == [1]


def test_warning_callback_not_invoked_when_allowing() -> None:
    calls = []
    otlp_filter = OTLPConnectionErrorFilter(warning_callback=lambda: calls.append(1))
    record = _make_record("some.other.logger", "unrelated message", level=logging.INFO)
    otlp_filter.filter(record)
    assert calls == []


def test_safe_logging_handler_emits_without_a_configured_provider() -> None:
    # No logger_provider given -> resolves the global (NoOp by default) provider.
    # emit() must not raise even though nothing is actually configured.
    handler = _SafeLoggingHandler()
    handler.emit(_make_record("test", "hello", level=logging.INFO))


def test_safe_logging_handler_recursion_guard_drops_reentrant_emit(monkeypatch) -> None:
    handler = _SafeLoggingHandler()
    calls = []

    def fake_super_emit(self: object, record: logging.LogRecord) -> None:
        calls.append(record.msg)
        # A reentrant call (e.g. the export path itself logging an error)
        # must be dropped by the guard instead of recursing.
        handler.emit(_make_record("reentrant", "should be dropped"))

    monkeypatch.setattr("opentelemetry.sdk._logs.LoggingHandler.emit", fake_super_emit)
    handler.emit(_make_record("test", "outer", level=logging.INFO))

    assert calls == ["outer"]
    # The guard must be released after the outer emit completes.
    assert _otel_handler_active.get() is False
