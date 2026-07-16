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
import io
import json
import logging

import pytest

from datarobot_genai.core.telemetry.logging import JsonFormatter
from datarobot_genai.core.telemetry.logging import ReadableFormatter
from datarobot_genai.core.telemetry.logging import RedactingFormatter
from datarobot_genai.core.telemetry.logging import TextFormatter
from datarobot_genai.core.telemetry.logging import get_logger
from datarobot_genai.core.telemetry.logging import init_logging


def _make_record(message: str = "hello", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_includes_extra_fields() -> None:
    record = _make_record("hello", user_id="u1")
    data = json.loads(JsonFormatter().format(record))
    assert data["message"] == "hello"
    assert data["user_id"] == "u1"
    assert data["level"] == "INFO"


def test_json_formatter_serializes_arbitrary_object_via_str_fallback() -> None:
    class Widget:
        def __str__(self) -> str:
            return "widget-repr"

    record = _make_record("hello", widget=Widget())
    data = json.loads(JsonFormatter().format(record))
    assert data["widget"] == "widget-repr"


def test_json_formatter_reports_circular_reference_error() -> None:
    circular: dict[str, object] = {}
    circular["self"] = circular
    record = _make_record("hello", circular=circular)
    data = json.loads(JsonFormatter().format(record))
    assert "serialization error" in data["circular"]


def test_text_formatter_appends_extra_fields() -> None:
    formatter = TextFormatter("%(message)s")
    record = _make_record("hello", user_id="u1")
    assert formatter.format(record) == "hello | user_id=u1"


def test_readable_formatter_single_line_without_exception() -> None:
    record = _make_record("hello")
    output = ReadableFormatter().format(record)
    assert output.endswith("INFO:test.logger:hello")
    assert "\n" not in output


def test_readable_formatter_indents_traceback() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _make_record("failed")
        record.exc_info = sys.exc_info()
    output = ReadableFormatter().format(record)
    assert "exception:" in output
    assert "ValueError: boom" in output


@pytest.mark.parametrize("sensitive_key", ["access_token", "refresh_token", "api_key"])
def test_redacting_formatter_redacts_direct_attribute(sensitive_key: str) -> None:
    record = _make_record("hello", **{sensitive_key: "super-secret"})
    formatted = RedactingFormatter(TextFormatter("%(message)s")).format(record)
    assert "super-secret" not in formatted
    assert "[REDACTED]" in formatted


def test_redacting_formatter_redacts_nested_dict() -> None:
    record = _make_record("hello", payload={"api_key": "super-secret", "ok": "value"})
    formatted = RedactingFormatter(TextFormatter("%(message)s")).format(record)
    assert "super-secret" not in formatted
    assert "value" in formatted


def test_redacting_formatter_regex_catches_string_representation() -> None:
    record = _make_record("request failed with api_key='super-secret'")
    formatted = RedactingFormatter(TextFormatter("%(message)s")).format(record)
    assert "super-secret" not in formatted


def test_redacting_formatter_does_not_mutate_original_record() -> None:
    record = _make_record("hello", api_key="super-secret")
    RedactingFormatter(TextFormatter("%(message)s")).format(record)
    assert record.api_key == "super-secret"


def test_init_logging_wires_redacting_formatter_by_default() -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    stream = io.StringIO()
    try:
        init_logging(stream=stream)
        logging.getLogger("test.init").info("secret api_key=super-secret here")
        output = stream.getvalue()
        assert "super-secret" not in output
        assert "[REDACTED]" in output
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for handler in original_handlers:
            root_logger.addHandler(handler)


def test_get_logger_readable_format() -> None:
    stream = io.StringIO()
    logger = get_logger("test.readable", format_type="readable", stream=stream)
    logger.info("hello")
    assert "INFO:test.readable:hello" in stream.getvalue()
