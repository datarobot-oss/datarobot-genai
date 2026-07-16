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

from datarobot_genai.core.telemetry.logging import JsonFormatter
from datarobot_genai.core.telemetry.logging import ReadableFormatter
from datarobot_genai.core.telemetry.logging import RedactingFormatter
from datarobot_genai.core.telemetry.uvicorn_filter import HealthCheckFilter
from datarobot_genai.core.telemetry.uvicorn_filter import configure_uvicorn_logging


def _make_record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_health_check_filter_suppresses_health_get_at_info() -> None:
    health_filter = HealthCheckFilter(log_level="INFO")
    record = _make_record('"GET /health HTTP/1.1" 200 OK')
    assert health_filter.filter(record) is False


def test_health_check_filter_allows_health_get_at_debug() -> None:
    health_filter = HealthCheckFilter(log_level="DEBUG")
    record = _make_record('"GET /health HTTP/1.1" 200 OK')
    assert health_filter.filter(record) is True


def test_health_check_filter_allows_other_requests() -> None:
    health_filter = HealthCheckFilter(log_level="INFO")
    record = _make_record('"GET /api/chats HTTP/1.1" 200 OK')
    assert health_filter.filter(record) is True


def test_configure_uvicorn_logging_json_format() -> None:
    configure_uvicorn_logging(log_format="json", log_level="INFO")
    access_logger = logging.getLogger("uvicorn.access")
    formatter = access_logger.handlers[0].formatter
    assert isinstance(formatter, RedactingFormatter)
    assert isinstance(formatter.original_formatter, JsonFormatter)
    assert any(isinstance(f, HealthCheckFilter) for f in access_logger.handlers[0].filters)


def test_configure_uvicorn_logging_readable_format() -> None:
    configure_uvicorn_logging(log_format="readable", log_level="INFO")
    error_logger = logging.getLogger("uvicorn.error")
    formatter = error_logger.handlers[0].formatter
    assert isinstance(formatter, RedactingFormatter)
    assert isinstance(formatter.original_formatter, ReadableFormatter)


def test_configure_uvicorn_logging_redacts_secrets_in_access_log() -> None:
    configure_uvicorn_logging(log_format="text", log_level="INFO")
    access_logger = logging.getLogger("uvicorn.access")
    record = _make_record('"GET /api/chats?api_key=super-secret HTTP/1.1" 200 OK')
    formatted = access_logger.handlers[0].formatter.format(record)
    assert "super-secret" not in formatted


def test_configure_uvicorn_logging_does_not_propagate() -> None:
    configure_uvicorn_logging(log_format="text", log_level="INFO")
    assert logging.getLogger("uvicorn.access").propagate is False
    assert logging.getLogger("uvicorn.error").propagate is False
