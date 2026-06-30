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

"""In-process mock OTLP/HTTP collector for e2e tests.

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

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from typing import Any

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

# OTLP/HTTP traces ingest path. The DataRobot collector ingress lives off
# ``/otel/v1/traces`` (see ``resolve_otel_traces_endpoint_from_env``).
OTLP_TRACES_PATH = "/otel/v1/traces"


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
